
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa

# Functions for Speech-to-Text
def speech_to_text_speechrecognition(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        return f"SpeechRecognition Error: {str(e)}"

def speech_to_text_wav2vec(audio_file):
    try:
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        audio_input, _ = librosa.load(audio_file, sr=16000)
        input_values = tokenizer(audio_input, return_tensors="pt", padding="longest").input_values

        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.decode(predicted_ids[0])
        return transcription
    except Exception as e:
        return f"Wav2Vec2 Error: {str(e)}"

# GUI Functions
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if file_path:
        audio_file_var.set(file_path)

def transcribe_speechrecognition():
    audio_file = audio_file_var.get()
    if not os.path.isfile(audio_file):
        messagebox.showerror("Error", "Please select a valid audio file!")
        return
    transcription = speech_to_text_speechrecognition(audio_file)
    display_output(transcription)

def transcribe_wav2vec():
    audio_file = audio_file_var.get()
    if not os.path.isfile(audio_file):
        messagebox.showerror("Error", "Please select a valid audio file!")
        return
    transcription = speech_to_text_wav2vec(audio_file)
    display_output(transcription)

def display_output(text):
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, text)

# GUI Setup
root = tk.Tk()
root.title("Speech Recognition System")

# File Selection
audio_file_var = tk.StringVar()
file_frame = tk.Frame(root)
file_frame.pack(pady=10)

tk.Label(file_frame, text="Audio File:").pack(side=tk.LEFT, padx=5)
file_entry = tk.Entry(file_frame, textvariable=audio_file_var, width=50)
file_entry.pack(side=tk.LEFT, padx=5)
file_button = tk.Button(file_frame, text="Browse", command=select_file)
file_button.pack(side=tk.LEFT, padx=5)

# Transcription Options
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

speechrec_button = tk.Button(button_frame, text="Transcribe (SpeechRecognition)", command=transcribe_speechrecognition)
speechrec_button.pack(side=tk.LEFT, padx=10)

wav2vec_button = tk.Button(button_frame, text="Transcribe (Wav2Vec2)", command=transcribe_wav2vec)
wav2vec_button.pack(side=tk.LEFT, padx=10)

# Output Display
output_frame = tk.Frame(root)
output_frame.pack(pady=10)

tk.Label(output_frame, text="Transcription Output:").pack(anchor=tk.W)
output_text = scrolledtext.ScrolledText(output_frame, width=60, height=15, wrap=tk.WORD)
output_text.pack(padx=5, pady=5)

# Run the Application
root.mainloop()
