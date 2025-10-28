# Transcripta
This is a project we made and submitted for our IDC409- Intro to DS and AI course. 

## Speech-to-Text and it's Applications

## Overview
This includes all the details of the project we have created for our **IDC409: Introduction to Data Science and Artificial Intelligence** course. The assignment was to take an audio file, transcribe its contents into text, and optionally use this text for further processing or text-to-speech synthesis. We developed a speech-to-text (STT) system utilising the `faster-whisper` library, which leverages OpenAI’s Whisper model.

This repository contains the **speech-to-text extraction module**, the first major step in the workflow.  
In addition to this core code, our project extends this skeleton into three advanced modules:
- **Multilingual Transcription and Language Detection:** automatically detecting and transcribing audio in multiple languages.  
- **Live Speech-to-Text Transcription:** enabling real-time transcription from microphone input.  
- **Model Comparison Framework:** comparing the accuracy and performance of different speech-to-text models.

These extensions demonstrate practical applications of speech recognition systems and explore how model choice, noise level and language diversity influence transcription accuracy.

---

### How It Works

The system utilises a **Whisper model** to transcribe a given audio file into text.  
The transcription process includes:
- Loading the Whisper model
- Reading the input audio file (like `.mp3` or `.wav`)
- Detecting language and generating text segments
- Printing and saving the transcribed text to a `.txt` file

The main idea is to provide an accurate and efficient transcription pipeline that can be easily integrated into larger AI/ML workflows.

---

### Code Explanation 

Below is an explanation of the script's main components.

```python
from faster_whisper import WhisperModel
import os, sys, datetime
```
Imports libraries:
- `faster_whisper` loads and runs the Whisper model
- `os, sys` handle file paths and command-line arguments
- `datetime` is used to timestamp transcript creation

```python
model_size = "small.en"
model = WhisperModel(model_size, device="cpu", compute_type="float32")
```
- Defines the Whisper model variant (`small.en` = English-only and efficient)
- Loads the model on CPU with 32-bit precision, suitable for local systems without GPU

```python
audio_arg = sys.argv[1] if len(sys.argv) > 1 else "audio.mp3"
```
- Allows the script to accept an audio file path as a command-line argument
- If no argument is provided, it defaults to `"audio.mp3"` in the same directory

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_path = audio_arg if os.path.isabs(audio_arg) else os.path.join(script_dir, audio_arg)
```
- This resolves absolute path of the audio file
- Ensures the file is correctly located relative to where the script is executed

```python
if not os.path.exists(audio_path):
    sys.exit(f"Audio file not found: {audio_path} ...")
```
- Performs the core transcription
- `segments` contain transcribed text chunks
- `info` holds metadata like detected language and confidence score
- `beam_size=5` improves accuracy by using beam search decoding

```python
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
```
- Displays detected language and confidence, useful for multilingual datasets

```python
transcript_lines = []
for segment in segments:
    print(segment.text)
    transcript_lines.append(segment.text)
```
- Iterates through all transcription segments, printing them to the console
- Stores each segment in a list for writing to a file later

```python
base = os.path.splitext(os.path.basename(audio_path))[0]
out_name = f"{base}_transcript.txt"
out_path = os.path.join(script_dir, out_name)
```
- Creates a new text file name based on the original audio name.\
Like: `audio.mp3` to `audio_transcript.txt`

```python
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"Source: {audio_path}\n")
    f.write(f"Model: {model_size}\n")
    f.write(f"Detected language: {info.language} (probability={info.language_probability})\n")
    f.write(f"Created: {datetime.datetime.now().isoformat()}\n\n")
    for line in transcript_lines:
        f.write(line.strip() + "\n")
```
- Writes metadata (source file, model, date) and transcription text to the output file
- UTF-8 encoding ensures compatibility with all characters

```python
print(f"Transcript saved to: {out_path}")
```
- Prints the output file location for user

---

### Running the Code 
1. Install libraries/ modules:
```bash
pip install faster-whisper
```
2. Place Your Audio File\
Save your audio file (like `audio.mp3`) in the same directory as the script/ note its full path.
3. Run the Script
You can run the script in two ways:
- Default audio file (`audio.mp3`)
```bash
python3 speech_to_text.py
```
- Custom audio file path
```bash
python3 speech_to_text.py /path/to/your_audio.mp3
```
---

## Extension 1: Multilingual Transcription and Language Detection

**Highlights/ features of this project are:**

- **Accurate Transcriptions:** Uses the Whisper medium model to capture nuances in speech and maintain context.

- **Multi-Format Audio Support:** Handles `.mp3`, `.wav`, `.m4a`, `.flac`, and more.

- **Automatic Audio Standardisation:** Optimises the audio files to ensure consistent transcription quality.

- **Language Detection:** Automatically identifies the spoken language and provides confidence scores.

- **Structured and Organised Outputs:** Generates individual transcript files with metadata and an aggregated CSV.

- **User-Friendly Interface:** A simple graphical file chooser enables easy selection of one or multiple audio files directly.

---

### Installation

This project requires **Python 3.10 or higher**. You also need to install some dependencies and `FFmpeg` for audio processing.

**Install dependencies:**
```python
pip install faster-whisper soundfile pandas matplotlib
```

**Install FFmpeg (platform-specific):**

macOS:
```python
brew install ffmpeg
```

Windows: Download the installer from `FFmpeg official website
 and follow the setup instructions.

Linux (Debian/Ubuntu):
```python
sudo apt update
sudo apt install ffmpeg
```

---

### Transcription Workflow
This is the step-by-step process this project uses to convert audio into accurate text, multilingually.

**Importing Libraries**\
Essential Python modules are loaded to handle system operations, file management, audio processing, GUI interaction, and the Whisper transcription model. Each module serves a specific purpose to support the end-to-end workflow.

:arrow_down:

**Model Configuration**\
The Whisper model is configured with parameters that balance speed and accuracy. Device selection and numerical precision are defined to ensure reliable transcription across different systems.

:arrow_down:

**Audio Conversion**\
Audio files are standardised to an optimal format for transcription. This includes converting to a single-channel WAV file, adjusting sample rates, and normalising audio levels. Temporary files are handled safely to prevent data loss.

:arrow_down:

**Transcription**\
Audio samples are passed through the Whisper model with parameters designed to maintain context and produce high-quality text. The transcription captures timestamps for words and detects the language automatically, with confidence scoring.

:arrow_down:

**File Management and Output**\
Each audio file is transcribed into a separate text file that includes metadata such as source, model, detected language, confidence, and timestamp. All individual transcripts are also aggregated into a CSV file for easy analysis and reporting.

:arrow_down:

**Graphical File Selection**\
A user-friendly GUI allows multiple audio files to be selected without requiring command-line input, simplifying the workflow for end users.

---

### **Importing Libraries**

We start by importing essential modules for system operations, audio handling, GUI, and the transcription model:

```python

import os
import sys
import datetime
import tempfile
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog
from faster_whisper import WhisperModel
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
```

Each module serves a precise role, from file management (os, shutil) to data handling (pandas) and transcription (faster-whisper).

---

### Model Configuration  

```python

model_size = "medium"
device = "cpu"
compute_type = "float32"
output_csv = "results.csv"
```
- `model_size` balances speed and accuracy.
- `device` ensures compatibility (CPU used here).
- `compute_type` ensures numerical precision.
- `output_csv` sets the default CSV file name.

### Script Directory Reference
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
```
We store outputs in the same directory as the script, ensuring paths are consistent across systems.

### Audio Conversion
Before transcription, audio files are converted to a standardised `.wav` format for optimal processing:

```python

def _convert_with_ffmpeg(src_path):
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
        try: os.remove(out_path)
        except Exception: pass
        raise RuntimeError("ffmpeg conversion failed for: " + src_path)
    return out_path
```
Errors are handled cleanly to prevent partial transcription or corrupted files.

### Transcription Function
```python
def transcribe_audio(audio_path, model):
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
            try: os.remove(use_tmp)
            except Exception: pass
```

### GUI Audio Picker

```python

def pick_audio_files():
    filetypes = [("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac"), ("All files", "*.*")]
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilenames(title="Select one or more audio files", initialdir=script_dir, filetypes=filetypes)
    root.update()
    root.destroy()
    return list(selected)
```


A graphical interface makes it easy to select multiple audio files without relying on the command line.

### Saving Transcripts and CSV

```python

base = os.path.splitext(os.path.basename(audio_arg))[0]
out_txt = os.path.join(script_dir, f"{base}_transcript.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(f"Source: {audio_arg}\n")
    f.write(f"Model: {model_size}\n")
    f.write(f"Language: {lang} (confidence: {prob})\n")
    f.write(f"Created: {datetime.datetime.now().isoformat()}\n\n")
    f.write(transcript)

rows.append({"File": os.path.basename(audio_arg), "Language": lang, "Confidence": prob, "Transcript": transcript})
df = pd.DataFrame(rows)
csv_path = os.path.join(script_dir, output_csv)
df.to_csv(csv_path, index=False)
```
All transcripts are aggregated into a `.csv` file, providing a complete dataset for analysis or reporting.

---

### Output

The transcription workflow delivers results in two complementary formats:

1. **Individual Transcript Files:**
- Each audio file produces a separate `.txt` file.
- Metadata: source filename, Whisper model version, detected language with confidence score, and timestamp.
- Provides the full, readable transcript for easy review.

2. **Aggregated CSV:**
- Compiles all transcripts into a single `.csv` file.
- Each row records the filename, language, confidence, and complete transcript.

---

## Extension 2: Live Speech-to-Text Transcription

This extension builds upon our base STT model by enabling **real-time transcription** of spoken input through a computer’s microphone.  
It continuously listens, detects active speech segments, and transcribes them into text, all without needing an audio file input.

### Key Idea

While the base code processes pre-recorded audio files, this script introduces a **live audio stream**.
The model captures sound data in small chunks, processes them through **voice activity detection (VAD)**, and runs transcription using `faster-whisper`.

This is especially useful for applications such as:
- Live meeting transcription
- Real-time captioning
- Voice command systems

### Components and Their Purpose

#### 1. Audio Input Stream
```python
import sounddevice as sd
samplerate = 16000
block_duration = 0.25
chunk_duration = 4.0 
channels = 1
```
The `sounddevice` library is used to capture live audio from the system’s microphone.\
The audio is recorded in small blocks (0.25 seconds each), which are queued for processing. We take longer chunks for better context.

#### 2. Queue and Threading 
```python
audio_queue = queue.Queue()
rec_thread = threading.Thread(target=recorder, daemon=True)
trans_thread = threading.Thread(target=transcriber, daemon=True)
```
Two threads run simultaneously:\
- **Recorder thread:** Listens to the microphone and pushes raw audio blocks into a queue
- **Transcriber thread:** Continuously consumes from the queue, processes speech, and generates transcriptions.\
This prevents blocking and ensures a continuous transcription.

#### 3. Voice Activity Detection (VAD)
```python
vad_threshold = 0.005 
class VAD:
    def is_speech(self, audio_data, base_threshold=0.005):
        energy = np.sqrt(np.mean(audio_data ** 2))
        ...
        return energy > dynamic_threshold
```
VAD only processes spoken audio, not silence or background noise.\
A dynamic threshold is used to adapt to varying ambient noise levels, improving accuracy and efficiency.

#### 4. Handling Overlaps
```python
overlap_duration = 0.85
context_buffer = []
context_size = 3
```
To maintain context and continuity between chunks, each new transcription chunk overlaps slightly with the previous one.\
The `context_buffer` ensures no text is duplicated and helps smooth out sentence boundaries.

#### 5. Filtering Similarities
```python
def similarity_score(text1, text2):
    words1 = set(text1.lower().split())
    ...
    return len(intersection) / shorter
```
This function prevents repeated phrases from being printed multiple times.\
If a new segment is too similar to recent ones, it’s skipped.

#### 6. Live Transcription Loop
```python
segments, _ = model.transcribe(
    chunk_mono,
    language="en",
    beam_size=5,
    condition_on_previous_text=False,
    vad_filter=True
)
```
Each 4-second chunk of speech is transcribed using the Whisper model.\
Parameters like `beam_size=5` control decoding accuracy, while `vad_filter=True` improves word boundary detection.

7. Output
```python
if transcript_lines:
    out_name = f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
```
All transcriptions are stored in a timestamped text file once the recording stops.\

---

### Running the Script
1. Dependencies:
Ensure the following Python libraries are installed:
```bash
pip install sounddevice numpy faster-whisper
```
2. To start live transcription, run:
```bash
python3 live_transcriber.py
```
3. Then speak into your microphone, the script will:
- Continuously listen to your voice.
- Transcribe what you say in real-time.
- Save the full transcript to a `.txt` file when stopped.\
Press **Enter** or **Ctrl+C** to stop recording.

---

## Extension 3: Speech Recognition Model Comparison Tool

### Overview

This module expands our project into a **comparative analysis tool** for evaluating multiple STT models on the same audio samples.  
It automatically benchmarks **Whisper**, **Vosk**, and **Google Speech Recognition**, comparing their:
- **Transcription accuracy** (WER, CER, Sentence & Punctuation Accuracy)
- **Speed and efficiency** (Real-Time Factor, RAM usage)
- **Noise robustness**
- **Overall performance through plots and tables**

A **GUI file selector** makes it simple to choose audio and reference transcript files, while automated plots and tables visualise performance across all models.

### Key Idea

Each STT engine uses a different architecture:
- **Whisper** | OpenAI | High accuracy, multilingual, robust 
- **Vosk** | Kaldi | Lightweight, offline, fast 
- **Google Speech Recognition** | Cloud API | High-quality but dependent on internet 

This script creates a **uniform evaluation pipeline:**
1. Converts and normalises audio files.  
2. Adds controlled background noise.  
3. Transcribes audio using all three models.  
4. Calculates multiple accuracy metrics.  
5. Plots and summarises results.

### Components and Explanations

#### 1. **Workspace Setup and Model Initialisation**
```python
workspace_dir = os.path.join(os.path.expanduser("~"), "speech_compare_workspace")
whisper_model = WhisperModel("small.en", device="cpu", compute_type="float32")
```
All results (plots, CSVs, and transcripts) are organised under a single workspace folder.\
The Whisper model is loaded once for efficiency.

#### 2. File Selection (GUI)
```python
from tkinter import filedialog
audio_files = filedialog.askopenfilenames(title="Select Audio Files")
```
A **Tkinter file** dialogue allows users to pick multiple audio and transcript files interactively, making the process beginner-friendly and avoiding manual path entry.

#### 3. Vosk Model Setup
```python
download_and_extract_vosk(vosk_zip_url, vosk_zip_path, workspace_dir)
vosk_model = VoskModel(vosk_dir)
```
If the Vosk model isn’t available, it’s automatically downloaded and extracted.\
Ensuring the script can run on any machine with internet access and no prior setup.

#### 4. Audio Preprocessing
```python
def convert_to_wav(src_path):
    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
```
Every file is standardised to mono **16 kHz WAV**, the preferred format for most STT models for fair comparison and consistent sampling rates.

#### 5. Noise Augmentation
```python
def add_noise(audio_path, snr_db=noise_level_db):
    noise = WhiteNoise().to_audio_segment(duration=len(audio))
    noisy = audio.overlay(noise)
```
Controlled background noise is added to simulate real-world recording conditions.\
The `snr_db` value determines 'how loud the noise is relative' to the voice.

#### 6. Performance Measurement
```python
def measure_performance(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, {"time_sec": end - start, "ram_mb": ram_end - ram_start}
```
Tracks execution time and RAM consumption for each transcription.\
This helps compare how efficiently each model runs under similar workloads.

#### 7. Metric Computation
```python
from jiwer import wer, cer
metrics['WER'] = wer(ref_norm, pred_norm)
metrics['CER'] = cer(ref_norm, pred_norm)
```
Four key accuracy metrics are computed:
- **WER (Word Error Rate)** measures word-level mistakes.
- **CER (Character Error Rate)** measures finer differences.
- **Sentence Accuracy** how many full sentences are transcribed correctly.
- **Punctuation Accuracy** evaluates preservation of punctuation marks.
Lower WER/CER and higher accuracy scores indicate better performance.

#### 8. Transcription Functions
Each model is wrapped in a separate transcription function:
```python
def transcribe_whisper(path): ...
def transcribe_vosk(path, model): ...
def transcribe_google(path): ...
```
This modular structure allows easy addition of future models (e.g., DeepSpeech or OpenAI Whisper large).

#### 9. Real-Time Factor (RTF)
```python
def compute_rtf(duration_sec, processing_time):
    return processing_time / duration_sec
```
RTF = 1: model processes at real-time speed
RTF < 1: faster than real-time (ideal)
RTF > 1: slower than real-time

#### 10. Data Aggregation and Visualisation
```python
df.to_csv(results_csv_path, index=False)
plot_results(df)
create_metrics_table(df)
```
After all transcriptions, results are saved to CSV and visualised using Matplotlib and Seaborn.
Example visualisations include:
- WER/CER bar charts
- Accuracy plots
- RTF comparisons
All plots are automatically stored in the `plots/` folder.

---

### Running the Script
1. Dependencies:
```bash
pip install faster-whisper vosk speechrecognition pydub jiwer psutil seaborn pandas matplotlib certifi
```
2. To launch the comparison tool:
```bash
python3 compare_models.py
```
Steps:
- Select multiple audio files (like, `.mp3`, `.wav`, `.flac`)
- Wait while each STT model transcribes the audio
- Review the printed performance tables, CSVs, and plots

---

## Conclusion
### Speech-to-Text Transcription Suite
A comprehensive Speech-to-Text suite implementing real-time streaming, batch processing, and comparative analysis of multiple STT libraries (Whisper, Vosk, Google Speech Recognition). The suite features dynamic Voice Activity Detection (VAD), multilingual support, and detailed performance metrics including WER, CER, and RTF analysis.

### Components:
- `Real-Time-Transcription.py`: Streaming transcription with VAD
- `Multilingual-STT.py`: Batch processing with language detection
- `Comparing-Libraries.py`: Performance analysis framework
- `Skeleton-Code.py`: Simple transcription template

### Performance:
- Whisper: Best accuracy, RTF < 1 (faster than real-time)
- Vosk: Reliable offline processing
- Google Speech: Good accuracy, requires internet
- Repository structure, documentation, and detailed usage examples available in individual component directories.

---

## Contributors
**Archisha Panda**\
**Kartik Dixit**\
**Sahil Deshmukh**

*Department of Biology*\
*Indian Institute of Science Education and Research, Mohali*
*2025*

---

## License
This project is for educational and research purposes only.\
All models and dependencies belong to their respective authors.

---
## Thank You :)
---

