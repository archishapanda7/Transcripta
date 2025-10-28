import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import sys
import os
import datetime
import math

# Configuration
samplerate = 16000
block_duration = 0.25
chunk_duration = 4.0         # longer chunks for better context
overlap_duration = 0.85      # more overlap to maintain context
channels = 1
vad_threshold = 0.005       # increased threshold to avoid background noise

frames_per_block = int(block_duration * samplerate)
frames_per_chunk = int(chunk_duration * samplerate)
frames_overlap = int(overlap_duration * samplerate)

# Load model - using small.en for better stability on CPU
model = WhisperModel("small.en", device="cpu", compute_type="float32")

# State management
audio_queue = queue.Queue()
audio_buffer = []
context_buffer = []          # store recent transcriptions
context_size = 3            # number of recent chunks to remember
transcript_lines = []
last_printed = ""
stop_event = threading.Event()

# Time-based dedupe state (avoid duplicates from overlapping chunks)
processed_frames = 0         # total frames already advanced (non-overlapping start)
last_segment_end_time = 0.0  # last accepted segment end (seconds)

# Improved VAD with dynamic threshold
class VAD:
    def __init__(self, window_size=50):
        self.energy_history = []
        self.window_size = window_size
        self.min_energy = float('inf')
        self.max_energy = 0
        
    def update_threshold(self, energy):
        self.energy_history.append(energy)
        if len(self.energy_history) > self.window_size:
            self.energy_history.pop(0)
        self.min_energy = min(energy, self.min_energy)
        self.max_energy = max(energy, self.max_energy)
    
    def is_speech(self, audio_data, base_threshold=0.005):
        energy = np.sqrt(np.mean(audio_data ** 2))
        self.update_threshold(energy)
        
        if len(self.energy_history) < 10:  # warmup period
            return energy > base_threshold
            
        ambient_energy = np.median(self.energy_history)
        dynamic_threshold = max(base_threshold, ambient_energy * 2.5)
        return energy > dynamic_threshold

vad = VAD()

def similarity_score(text1, text2):
    """Compare text similarity based on word overlap"""
    if not text1 or not text2:
        return 0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    shorter = min(len(words1), len(words2))
    if shorter == 0:
        return 0
    return len(intersection) / shorter

def is_unique_text(new_text, context, threshold=0.7):
    """Check if text is sufficiently different from context"""
    if not new_text or not context:
        return True
    return all(similarity_score(new_text, prev) < threshold for prev in context)

def audio_callback(indata, frames, time, status):
    if status:
        print("InputStream status:", status)
    audio_queue.put(indata.copy())

def recorder():
    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32',
                          callback=audio_callback, blocksize=frames_per_block):
            print("Listening... press Enter or Ctrl+C to stop")
            while not stop_event.is_set():
                sd.sleep(100)
    except Exception as e:
        print("Recorder error:", e)
        stop_event.set()

def _seg_time(segment, key):
    # helper to read start/end from segment (works with object or dict)
    if hasattr(segment, key):
        return getattr(segment, key)
    if isinstance(segment, dict):
        return segment.get(key, 0.0)
    return 0.0

def transcriber():
    global audio_buffer, context_buffer, transcript_lines, processed_frames, last_segment_end_time
    
    # Warmup period
    print("Calibrating ambient noise...")
    for _ in range(8):  # longer warmup
        try:
            block = audio_queue.get(timeout=0.5)
            vad.is_speech(block.flatten())  # calibrate VAD
        except queue.Empty:
            continue
    
    while not stop_event.is_set():
        try:
            block = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)
        
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)
            chunk = audio_data[:frames_per_chunk]
            leftover = audio_data[frames_per_chunk - frames_overlap:]
            audio_buffer = [leftover] if len(leftover) > 0 else []
            
            # absolute start frame for this chunk (used to compute absolute timestamps)
            chunk_start_frame = processed_frames
            # advance processed_frames by the non-overlapping step
            processed_frames += (frames_per_chunk - frames_overlap)
            
            chunk_mono = chunk.flatten().astype(np.float32)
            if not vad.is_speech(chunk_mono):
                continue

            try:
                segments, _ = model.transcribe(
                    chunk_mono,
                    language="en",
                    beam_size=5,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=700,
                        speech_pad_ms=400
                    )
                )
                
                for segment in segments:
                    text = segment.text.strip() if hasattr(segment, "text") else (segment.get("text", "").strip() if isinstance(segment, dict) else "")
                    if not text:
                        continue
                        
                    # compute absolute segment start/end (seconds)
                    seg_start = _seg_time(segment, "start")
                    seg_end = _seg_time(segment, "end")
                    abs_start = (chunk_start_frame / samplerate) + (seg_start or 0.0)
                    abs_end = (chunk_start_frame / samplerate) + (seg_end or abs_start)
                    
                    # time-based dedupe: skip segments overlapping previously accepted audio
                    tol = 0.15  # seconds tolerance to avoid micro-overlaps
                    if abs_end <= last_segment_end_time + 1e-6 or abs_start < last_segment_end_time - tol:
                        continue

                    if is_unique_text(text, context_buffer):
                        print(text)
                        transcript_lines.append(text)
                        context_buffer.append(text)
                        if len(context_buffer) > context_size:
                            context_buffer.pop(0)
                        # update last accepted end time
                        last_segment_end_time = max(last_segment_end_time, abs_end)
            except Exception as e:
                print("Transcription error:", e)

    # Process remaining audio
    if audio_buffer:
        final_audio = np.concatenate(audio_buffer)
        chunk_mono = final_audio.flatten().astype(np.float32)
        if vad.is_speech(chunk_mono):
            try:
                # chunk_start_frame is current processed_frames (no further overlap advance)
                chunk_start_frame = processed_frames
                segments, _ = model.transcribe(
                    chunk_mono,
                    language="en",
                    beam_size=5,
                    condition_on_previous_text=False,
                    vad_filter=True
                )
                for segment in segments:
                    text = segment.text.strip() if hasattr(segment, "text") else (segment.get("text", "").strip() if isinstance(segment, dict) else "")
                    if not text:
                        continue
                    seg_start = _seg_time(segment, "start")
                    seg_end = _seg_time(segment, "end")
                    abs_start = (chunk_start_frame / samplerate) + (seg_start or 0.0)
                    abs_end = (chunk_start_frame / samplerate) + (seg_end or abs_start)
                    tol = 0.15
                    if abs_end <= last_segment_end_time + 1e-6 or abs_start < last_segment_end_time - tol:
                        continue
                    if text and is_unique_text(text, context_buffer):
                        print(text)
                        transcript_lines.append(text)
                        last_segment_end_time = max(last_segment_end_time, abs_end)
            except Exception as e:
                print("Final transcription error:", e)

# Start threads
rec_thread = threading.Thread(target=recorder, daemon=True)
trans_thread = threading.Thread(target=transcriber, daemon=True)
rec_thread.start()
trans_thread.start()

try:
    input("Press Enter to stop recording...\n")
except KeyboardInterrupt:
    pass

stop_event.set()
rec_thread.join(timeout=2)
trans_thread.join(timeout=2)
print("Stopped.")

if transcript_lines:
    out_name = f"transcript_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Created: {datetime.datetime.now().isoformat()}\n\n")
            for line in transcript_lines:
                f.write(line.strip() + "\n")
        print("Transcript saved to:", out_path)
    except Exception as e:
        print("Failed to save transcript:", e)