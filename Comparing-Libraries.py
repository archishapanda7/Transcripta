import os, time, psutil, wave, json, string, ssl, zipfile
import tkinter as tk
from tkinter import filedialog
import certifi
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from faster_whisper import WhisperModel
from vosk import Model as VoskModel, KaldiRecognizer
import speech_recognition as sr
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, RemoveWhiteSpace

#  Global CONFIGuRATION 
workspace_dir = os.path.join(os.path.expanduser("~"), "speech_compare_workspace")
os.makedirs(workspace_dir, exist_ok=True)

model_size = "small.en"
whisper_model = WhisperModel(model_size, device="cpu", compute_type="float32")

vosk_zip_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
vosk_zip_path = os.path.join(workspace_dir, "vosk-model.zip")

noise_level_db = 10
results_csv_path = os.path.join(workspace_dir, "comparison_results.csv")
plots_folder = os.path.join(workspace_dir, "plots")
os.makedirs(plots_folder, exist_ok=True)

def choose_files(title, filetypes):
    """Show file picker dialog and return selected paths."""
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    return list(paths)

def download_and_extract_vosk(url, zip_path, extract_to):
    """Download and extract Vosk model if not present."""
    for root, dirs, _ in os.walk(extract_to):
        if any(d.lower().startswith("vosk-model") for d in dirs):
            return
    
    print("Downloading Vosk model...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(url, context=ctx) as resp, open(zip_path, "wb") as out:
            while True:
                chunk = resp.read(16*1024)
                if not chunk:
                    break
                out.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        os.remove(zip_path)
    except Exception as e:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass
        raise RuntimeError(f"Vosk model download/extract failed: {e}")

def find_vosk_model(work_dir):
    """Locate extracted Vosk model folder."""
    for root, dirs, _ in os.walk(work_dir):
        for d in dirs:
            if d.lower().startswith("vosk-model"):
                return os.path.join(root, d)
    return None

def convert_to_wav(src_path, out_dir=workspace_dir):
    """Convert audio to 16kHz mono WAV."""
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(out_dir, f"{base}.wav")
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(out_path, format="wav")
    return out_path, len(audio)/1000.0  # Return path and duration in seconds

def add_noise(audio_path, snr_db=noise_level_db):
    """Create noisy version of audio file."""
    audio = AudioSegment.from_file(audio_path)
    noise = WhiteNoise().to_audio_segment(duration=len(audio))
    noise = noise - (noise.dBFS - audio.dBFS - snr_db)
    noisy = audio.overlay(noise)
    base = os.path.splitext(audio_path)[0]
    noisy_path = f"{base}_noisy.wav"
    noisy.export(noisy_path, format="wav")
    return noisy_path

def compute_rtf(duration_sec, processing_time):
    """Compute Real-Time Factor."""
    return processing_time / duration_sec if duration_sec > 0 else None

def measure_performance(func, *args):
    """Measure execution time and RAM usage of function."""
    process = psutil.Process(os.getpid())
    ram_start = process.memory_info().rss / 1024 / 1024
    start = time.time()
    result = func(*args)
    end = time.time()
    ram_end = process.memory_info().rss / 1024 / 1024
    return result, {"time_sec": end - start, "ram_mb": ram_end - ram_start}

def compute_metrics(predicted, reference):
    """Compute core metrics."""
    if not reference:
        return {}
    transform = Compose([RemovePunctuation(), RemoveWhiteSpace(), ToLowerCase()])
    pred_norm = transform(predicted)
    ref_norm = transform(reference)
    
    metrics = {}
    try:
        metrics['WER'] = wer(ref_norm, pred_norm)
        metrics['CER'] = cer(ref_norm, pred_norm)
    except Exception:
        metrics['WER'] = metrics['CER'] = None
    
    # Sentence accuracy
    pred_sents = [s.strip() for s in predicted.split('.') if s.strip()]
    ref_sents = [s.strip() for s in reference.split('.') if s.strip()]
    metrics['sentence_accuracy'] = sum(1 for p,r in zip(pred_sents, ref_sents) if p==r) / max(len(ref_sents),1)
    
    # Punctuation accuracy
    ref_punct = [c for c in reference if c in string.punctuation]
    pred_punct = [c for c in predicted if c in string.punctuation]
    metrics['punctuation_accuracy'] = (sum(1 for r,p in zip(ref_punct,pred_punct) if r==p) / max(len(ref_punct),1)) if ref_punct else 1.0
    
    return metrics

def transcribe_whisper(path):
    """Transcribe with Whisper."""
    segments, _ = whisper_model.transcribe(path, language="en", beam_size=5)
    return " ".join([s.text.strip() for s in segments])

def transcribe_vosk(path, model):
    """Transcribe with Vosk."""
    wf = wave.open(path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    text = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result.get("text"):
                text.append(result["text"])
    final = json.loads(rec.FinalResult()).get("text", "")
    if final:
        text.append(final)
    wf.close()
    return " ".join(text)

def transcribe_google(path):
    """Transcribe with Google Speech Recognition."""
    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio = r.record(source)
    return r.recognize_google(audio)

def plot_results(df):
    """Create bar plots for core metrics and RTF."""
    metrics = ['WER', 'CER', 'sentence_accuracy', 'punctuation_accuracy', 'RTF']
    
    for metric in metrics:
        if metric not in df.columns or df[metric].dropna().empty:
            continue
        plt.figure(figsize=(8,5))
        
        if metric == 'RTF':
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, 
                       label='Real-time threshold')
        
        sns.barplot(x='Library', y=metric, data=df, ci='sd', palette='Set2')
        plt.title(f"{metric} comparison")
        
        if metric == 'RTF':
            plt.ylabel('Real-Time Factor (lower is better)')
            plt.legend()
        
        plt.savefig(os.path.join(plots_folder, f"{metric}_bar.png"), 
                    bbox_inches='tight')
        plt.close()

def create_metrics_table(df):
    """Create overall metrics comparison table."""
    metrics = ['WER', 'CER', 'sentence_accuracy', 'punctuation_accuracy', 'RTF']
    summary = df.groupby('Library')[metrics].agg(['mean', 'std']).round(3)
    
    headers = ['Library', 'WER', 'CER', 'Sent. Acc', 'Punct. Acc', 'RTF']
    lib_width = max(len('Library'), max(len(lib) for lib in summary.index))
    metric_width = 12
    
    hline = f"+{'-'*(lib_width+2)}+" + f"{'-'*(metric_width+2)}+"*5
    
    print("\nOverall Performance Metrics (mean ± std)")
    print(hline)
    print(f"|{'Library':^{lib_width}}|" + "".join(f"{h:^{metric_width}}|" for h in headers[1:]))
    print(hline)
    
    for lib in summary.index:
        row = [f"{lib:<{lib_width}}"]
        for metric in metrics:
            mean = summary.loc[lib, (metric, 'mean')]
            std = summary.loc[lib, (metric, 'std')]
            row.append(f"{mean:.3f}±{std:.3f}")
        print("|" + "|".join(f"{val:^{metric_width}}" for val in row) + "|")
    
    print(hline)
    print("\nInterpretations:")
    print("- WER/CER: Lower is better (error rates)")
    print("- Sentence/Punctuation Accuracy: Higher is better")
    print("- RTF < 1: Faster than real-time, RTF > 1: Slower than real-time")

def create_rtf_table(df):
    """Create per-sample RTF comparison table."""
    rtf_pivot = df.pivot(index='Audio', columns='Library', values='RTF').round(3)
    
    headers = ['Audio File'] + list(rtf_pivot.columns)
    file_width = max(len('Audio File'), max(len(f) for f in rtf_pivot.index))
    lib_width = 12
    
    hline = f"+{'-'*(file_width+2)}+" + f"{'-'*(lib_width+2)}+"*len(rtf_pivot.columns)
    
    print("\nPer-Sample RTF Values (lower is better)")
    print(hline)
    print(f"|{'Audio File':^{file_width}}|" + "".join(f"{h:^{lib_width}}|" for h in rtf_pivot.columns))
    print(hline)
    
    for idx in rtf_pivot.index:
        row = [f"{idx:<{file_width}}"]
        for lib in rtf_pivot.columns:
            val = rtf_pivot.loc[idx, lib]
            row.append(f"{val:.3f}")
        print("|" + "|".join(f"{val:^{lib_width}}" for val in row) + "|")
    
    print(hline)
    print("\n* RTF < 1: Faster than real-time")
    print("* RTF = 1: Real-time")
    print("* RTF > 1: Slower than real-time")

def main():
    # Select files
    print("Select audio files:")
    audio_files = choose_files("Select Audio Files", 
                             [("Audio Files", "*.mp3 *.m4a *.wav *.flac *.ogg")])
    if not audio_files:
        print("No audio files selected.")
        return

    print("Select reference transcripts (optional):")
    ref_files = choose_files("Select Reference Transcripts", 
                           [("Text Files", "*.txt")])
    
    # Setup Vosk
    try:
        download_and_extract_vosk(vosk_zip_url, vosk_zip_path, workspace_dir)
    except Exception as e:
        print(f"Vosk setup failed: {e}")
        return

    vosk_dir = find_vosk_model(workspace_dir)
    if not vosk_dir:
        print("Vosk model not found.")
        return
    
    vosk_model = VoskModel(vosk_dir)
    
    # Process files
    results = []
    for audio in audio_files:
        print(f"\nProcessing: {os.path.basename(audio)}")
        wav, duration = convert_to_wav(audio)
        noisy = add_noise(wav)
        
        # Find matching reference
        ref_text = ""
        if ref_files:
            audio_base = os.path.splitext(os.path.basename(audio))[0].lower()
            for ref in ref_files:
                ref_base = os.path.splitext(os.path.basename(ref))[0].lower()
                if ref_base in audio_base or audio_base in ref_base:
                    with open(ref, 'r') as f:
                        ref_text = f.read().strip()
                    break
        
        # Run transcriptions
        for name, func in [
            ("Whisper", lambda: transcribe_whisper(wav)),
            ("Vosk", lambda: transcribe_vosk(wav, vosk_model)),
            ("Google", lambda: transcribe_google(wav))
        ]:
            try:
                transcript, perf = measure_performance(func)
                metrics = compute_metrics(transcript, ref_text) if ref_text else {}
                rtf = compute_rtf(duration, perf["time_sec"])
                
                results.append({
                    "Audio": os.path.basename(audio),
                    "Library": name,
                    "Time(s)": perf["time_sec"],
                    "RAM(MB)": perf["ram_mb"],
                    "RTF": rtf,
                    "Transcript": transcript,
                    **metrics
                })
                
                print(f"\n{name} completed")
                
            except Exception as e:
                print(f"Error with {name}: {e}")
    
    # Generate outputs
    if results:
        df = pd.DataFrame(results)
        df.to_csv(results_csv_path, index=False)
        plot_results(df)
        create_metrics_table(df)
        create_rtf_table(df)
        print(f"\nResults saved to: {results_csv_path}")
        print(f"Plots saved in: {plots_folder}")

if __name__ == "__main__":
    main()