# from flask import Flask,current_app, request, redirect, url_for, render_template, send_from_directory
# import torch
# import torchaudio
# from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
# import os
# import librosa
# import tempfile
# from zipfile import ZipFile
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, current_app
from werkzeug.utils import secure_filename
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import librosa
import tempfile
import matplotlib
matplotlib.use('Agg')  # Must be called before importing plt
import matplotlib.pyplot as plt
import librosa.display

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Initialize the model
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return redirect(url_for('separate', filename=file.filename))

# @app.route('/separate/<filename>')
# def separate(filename):
#     try:
#         input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         if not os.path.exists(input_path):
#             current_app.logger.error(f"File not found: {input_path}")
#             return "File not found", 404

#         waveform, sr = librosa.load(input_path, sr=None, mono=False)
#         waveform = torch.tensor(waveform, device=device).unsqueeze(0)
#         waveform = (waveform - waveform.mean()) / waveform.std()

#         separated = separate_sources(model, waveform, sr)
#         if separated is None:
#             raise ValueError("Failed to separate sources")

#         # Folder to save the separated sources
#         output_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}_stem')
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Save the separated stems as individual files
#         sources_list = ['drums', 'bass', 'other', 'vocals']
#         for i, source in enumerate(sources_list):
#             output_path = os.path.join(output_dir, f"{source}_{filename}")
#             torchaudio.save(output_path, separated[0][i].cpu(), sr)

#         # Zip the folder containing all stems
#         zip_filename = f"{filename}_stem.zip"
#         zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_filename)
#         with ZipFile(zip_path, 'w') as zipf:
#             for stem_file in os.listdir(output_dir):
#                 zipf.write(os.path.join(output_dir, stem_file), stem_file)

#         # Clean up the separated files directory
#         for stem_file in os.listdir(output_dir):
#             os.remove(os.path.join(output_dir, stem_file))
#         os.rmdir(output_dir)
        
#         # Send the zip file
#         return send_file(zip_path, as_attachment=True)

#     except Exception as e:
#         current_app.logger.error(f"Error during separation: {e}")
#         return "Error processing file", 500

@app.route('/separate/<filename>')
def separate(filename):
    try:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        waveform, sr = librosa.load(input_path, sr=None, mono=False)
        waveform = torch.tensor(waveform, device=device).unsqueeze(0)
        waveform = (waveform - waveform.mean()) / waveform.std()
        separated = separate_sources(model, waveform, sr)
        if separated is None:
            raise ValueError("Failed to separate sources")

        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'separated_sources')
        os.makedirs(output_dir, exist_ok=True)
        spectrogram_files = {}
        sources_list = ['drums', 'bass', 'other', 'vocals']
        for i, source in enumerate(sources_list):
            output_path = os.path.join(output_dir, f"{source}_{secure_filename(filename)}")
            torchaudio.save(output_path, separated[0][i].cpu(), sr)
            
            # Generate and save the spectrogram
            spectrogram_path = os.path.join(output_dir, f"{source}_{filename}.png")
            save_spectrogram(separated[0][i], sr, spectrogram_path)
            spectrogram_files[source.capitalize()] = url_for('download_file', filename=f"{source}_{filename}.png")


        # Generate the URLs for the separated tracks
        separated_files = {
            'Drums': url_for('download_file', filename=f"drums_{secure_filename(filename)}"),
            'Bass': url_for('download_file', filename=f"bass_{secure_filename(filename)}"),
            'Other': url_for('download_file', filename=f"other_{secure_filename(filename)}"),
            'Vocals': url_for('download_file', filename=f"vocals_{secure_filename(filename)}")
        }

        # Render the template with the audio players
        return render_template('index.html', separated_files=separated_files,spectrogram_files=spectrogram_files)

    except Exception as e:
        current_app.logger.error(f"Error during separation: {e}")
        return "Error processing file", 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'separated_sources'), filename)

def send_file(path, as_attachment=False):
    return send_from_directory(os.path.dirname(path), os.path.basename(path), as_attachment=as_attachment)


# Function for source separation
def separate_sources(model, mix, sample_rate, segment=10.0, overlap=0.1, progress_callback=None):
    from torchaudio.transforms import Fade
    chunk_len = int(sample_rate * segment * (1 + overlap))
    # Ensure there's at least one chunk
    total_chunks = max(1, mix.shape[2] // (chunk_len - int(overlap * sample_rate)))
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap * sample_rate), fade_shape="linear")
    final = torch.zeros(mix.shape[0], 4, mix.shape[1], mix.shape[2], device=device)

    for chunk_idx in range(total_chunks + 1):
        start = chunk_idx * (chunk_len - int(overlap * sample_rate))
        end = start + chunk_len
        if start >= mix.shape[2]:
            break
        if end > mix.shape[2]:
            end = mix.shape[2]
        chunk = mix[:, :, start:end]
        with torch.no_grad(): 
            out = model(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out

        # Update the progress bar
        if progress_callback is not None:
            progress_callback(chunk_idx / total_chunks)

    return final


def save_spectrogram(tensor, sr, filepath):
    fig, ax = plt.subplots()
    S = librosa.feature.melspectrogram(y=tensor.numpy()[0], sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Mel-frequency spectrogram')
    fig.savefig(filepath)
    plt.close(fig)



if __name__ == '__main__':
    app.run(debug=True)
