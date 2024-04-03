# import torch
# import torchaudio
# from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
# from torchaudio.utils import download_asset
# import os
# import librosa


# # Load the H-Demucs model
# bundle = HDEMUCS_HIGH_MUSDB_PLUS
# model = bundle.get_model()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Define a function to perform source separation
# def separate_sources(model, mix, sample_rate, segment=10.0, overlap=0.1):
#     from torchaudio.transforms import Fade
#     chunk_len = int(sample_rate * segment * (1 + overlap))
#     start = 0
#     end = chunk_len
#     overlap_frames = overlap * sample_rate
#     fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")
#     final = torch.zeros(mix.shape[0], len(model.sources), mix.shape[1], mix.shape[2], device=device)

#     while start < mix.shape[2] - overlap_frames:
#         chunk = mix[:, :, start:end]
#         with torch.no_grad():
#             out = model.forward(chunk)
#         out = fade(out)
#         final[:, :, :, start:end] += out
#         if start == 0:
#             fade.fade_in_len = int(overlap_frames)
#             start += int(chunk_len - overlap_frames)
#         else:
#             start += chunk_len
#         end += chunk_len
#         if end >= mix.shape[2]:
#             fade.fade_out_len = 0
#     return final

# # Example usage with an input file
# input_file = 'alone.wav' # Update this with your file path

# # Load the audio file with librosa
# waveform, sr = librosa.load(input_file, sr=None, mono=False)
# waveform = torch.tensor(waveform).unsqueeze(0)  # Convert to tensor and add batch dimension


# # Normalize and separate sources
# waveform = (waveform - waveform.mean()) / waveform.std()
# separated = separate_sources(model, waveform, sample_rate=sr, segment=10.0, overlap=0.1)


# # Save the separated sources
# output_dir = 'separated_sources'
# os.makedirs(output_dir, exist_ok=True)
# sources_list = ['drums', 'bass', 'other', 'vocals']
# for i, source in enumerate(sources_list):
#     torchaudio.save(os.path.join(output_dir, f"{source}.wav"), separated[0][i].cpu(), sr)


import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import os
import librosa
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
import threading

# Initialize the model outside of the GUI class
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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


class SeparationApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Music Source Separation')
        self.geometry('600x400')

        # Add the text - Model name and description
        self.model_name_label = ctk.CTkLabel(self, text="Model: H-DEMUCS High Quality (MusDB+)")
        self.model_name_label.pack(pady=20)

        self.file_path_entry = ctk.CTkEntry(self, placeholder_text="No file selected")
        self.file_path_entry.pack(pady=20)

        self.browse_button = ctk.CTkButton(self, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=10)

        self.separate_button = ctk.CTkButton(self, text="Separate", command=self.start_separation_thread)
        self.separate_button.pack()

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(pady=20)

        self.status_label = ctk.CTkLabel(self, text="Status: Waiting")
        self.status_label.pack(pady=20)
        self.progress_bar.set(0)

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        self.file_path_entry.delete(0, tk.END)
        self.file_path_entry.insert(0, file_path)

    def start_separation_thread(self):
        self.separate_thread = threading.Thread(target=self.separate_sources, daemon=True)
        self.separate_thread.start()

    def separate_sources(self):
        input_file = self.file_path_entry.get()
        self.status_label.configure(text="Status: Processing...")
        self.progress_bar.set(0)

        waveform, sr = librosa.load(input_file, sr=None, mono=False)
        waveform = torch.tensor(waveform, device=device).unsqueeze(0)
        waveform = (waveform - waveform.mean()) / waveform.std()

        def update_progress(progress):
            self.progress_bar.set(progress)
            if progress >= 1.0:
                self.status_label.configure(text="Status: Done. Check separated_sources folder.")

        separated = separate_sources(model, waveform, sample_rate=sr, segment=10.0, overlap=0.1,
                                     progress_callback=update_progress)

        output_dir = 'separated_sources'
        os.makedirs(output_dir, exist_ok=True)
        sources_list = ['drums', 'bass', 'other', 'vocals']
        for i, source in enumerate(sources_list):
            torchaudio.save(os.path.join(output_dir, f"{source}.wav"), separated[0][i].cpu(), sr)

if __name__ == "__main__":
    app = SeparationApp()
    app.mainloop()
