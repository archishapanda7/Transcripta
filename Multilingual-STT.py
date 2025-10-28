import os              # Operating system interface
import sys            # System-specific parameters and functions
import datetime       # Date and time handling
import tempfile       # Generate temporary files
import subprocess     # Spawn new processes, connect to their pipes
import shutil        # High-level file operations
import tkinter as tk  # GUI toolkit for file picker
from tkinter import filedialog  # file dialogue
from faster_whisper import WhisperModel  # Speech recognition model
import soundfile as sf    # Audio file handling
import pandas as pd      # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting library

# Model Configuration Parameters
model_size = "medium"      # Whisper model size (larger = more accurate but slower)
device = "cpu"            # Processing device (cpu/cuda)
compute_type = "float32"  # Numerical precision for computations
output_csv = "results.csv"  # Default CSV output filename

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def _convert_with_ffmpeg(src_path):
    """
    Convert input audio to optimal format for Whisper processing
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it with: brew install ffmpeg")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ar", "16000",
        "-ac", "1",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        out_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if res.returncode != 0 or not os.path.exists(out_path):
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise RuntimeError("ffmpeg conversion failed for: " + src_path)
    return out_path

def transcribe_audio(audio_path, model):
    """
    Process audio file through Whisper model for transcription
    """
    use_tmp = None
    try:
        use_tmp = _convert_with_ffmpeg(audio_path)
        samples, rate = sf.read(use_tmp, dtype="float32")
        segments, info = model.transcribe(
            samples,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            initial_prompt="The following is a high-quality transcription:"
        )
        transcript = " ".join([seg.text.strip() for seg in segments]).strip()
        return info.language, info.language_probability, transcript
    finally:
        if use_tmp:
            try:
                os.remove(use_tmp)
            except Exception:
                pass

def pick_audio_files():
    """
    Use a GUI file picker to select one or more audio files.
    """
    filetypes = [
        ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac"),
        ("All files", "*.*")
    ]
    root = tk.Tk()
    root.withdraw()
    # allow multiple selection
    selected = filedialog.askopenfilenames(title="Select one or more audio files", initialdir=script_dir, filetypes=filetypes)
    root.update()
    root.destroy()
    return list(selected)

def main():
    """Main execution: pick files, transcribe each, save per-file TXT and aggregated CSV."""
    audio_files = pick_audio_files()
    if not audio_files:
        sys.exit("No audio files selected.")
    print(f"Loading Whisper model ({model_size})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    rows = []
    for audio_arg in audio_files:
        print(f"\nTranscribing: {os.path.basename(audio_arg)}")
        try:
            lang, prob, transcript = transcribe_audio(audio_arg, model)
        except Exception as e:
            print(f"Error transcribing {audio_arg}: {e}")
            lang, prob, transcript = None, None, ""
        print(f"Detected Language: {lang} (confidence: {prob})")
        print("Transcript preview:", (transcript[:200] + "...") if len(transcript) > 200 else transcript)

        # Save transcript to text file
        base = os.path.splitext(os.path.basename(audio_arg))[0]
        out_txt = os.path.join(script_dir, f"{base}_transcript.txt")
        try:
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(f"Source: {audio_arg}\n")
                f.write(f"Model: {model_size}\n")
                f.write(f"Language: {lang} (confidence: {prob})\n")
                f.write(f"Created: {datetime.datetime.now().isoformat()}\n\n")
                f.write(transcript)
            print(f"Transcript saved to: {out_txt}")
        except Exception as e:
            print(f"Failed to save transcript for {audio_arg}: {e}")

        rows.append({
            "File": os.path.basename(audio_arg),
            "Language": lang,
            "Confidence": prob,
            "Transcript": transcript
        })

    # Save aggregated CSV
    try:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(script_dir, output_csv)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    except Exception as e:
        print("Failed to save CSV:", e)

if __name__ == "__main__":
    main()