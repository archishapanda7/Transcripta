from faster_whisper import WhisperModel
import os, sys, datetime

model_size = "small.en"
model = WhisperModel(model_size, device="cpu", compute_type="float32")

audio_arg = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = audio_arg if os.path.isabs(audio_arg) else os.path.join(script_dir, audio_arg)

if not os.path.exists(audio_path):
    sys.exit(f"Audio file not found: {audio_path}\nPut the file there, or run:\n  python3 {os.path.abspath(__file__)} /path/to/audio2.mp3")

segments, info = model.transcribe(audio_path, language="en", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

transcript_lines = []
for segment in segments:
    print(segment.text)
    transcript_lines.append(segment.text)

# write transcript file next to the script, named after the audio file
base = os.path.splitext(os.path.basename(audio_path))[0]
out_name = f"{base}_transcript.txt"
out_path = os.path.join(script_dir, out_name)
try:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {audio_path}\n")
        f.write(f"Model: {model_size}\n")
        f.write(f"Detected language: {info.language} (probability={info.language_probability})\n")
        f.write(f"Created: {datetime.datetime.now().isoformat()}\n\n")
        for line in transcript_lines:
            f.write(line.strip() + "\n")
    print(f"Transcript saved to: {out_path}")
except Exception as e:
    print("Failed to write transcript file:", e)