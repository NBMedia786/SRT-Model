import collections
import webrtcvad
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
import numpy as np
import torch
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Core libraries

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Advanced audio preprocessing for optimal Whisper performance"""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3) -----------------------------------------------------------------------------------------------------------

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to optimal range"""
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms * 0.1

        # Peak normalization as backup
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9

        return audio.astype(np.float32)

    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Advanced noise reduction using spectral subtraction"""
        # Convert to frequency domain
        D = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(D)
        phase = np.angle(D)

        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * sr / 512)
        noise_magnitude = np.mean(
            magnitude[:, :noise_frames], axis=1, keepdims=True)

        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        magnitude_clean = magnitude - alpha * noise_magnitude
        magnitude_clean = np.maximum(magnitude_clean, 0.1 * magnitude)

        # Reconstruct signal
        D_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(D_clean, hop_length=512)

        return audio_clean

    # def enhance_speech(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Enhance speech characteristics"""
        # High-pass filter to remove low-frequency noise
        sos = signal.butter(5, 80, btype='high', fs=sr, output='sos')
        audio = signal.sosfilt(sos, audio)

        # Gentle compression to balance dynamic range
        audio = np.sign(audio) * np.power(np.abs(audio), 0.8)

        return audio

    def apply_vad(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """Apply Voice Activity Detection"""
        # Convert to 16-bit PCM for VAD
        audio_16bit = (audio * 32767).astype(np.int16)

        # Frame settings for VAD
        frame_duration = 30  # ms
        frame_length = int(sr * frame_duration / 1000)

        voiced_frames = []
        speech_segments = []

        # Process in frames
        for i in range(0, len(audio_16bit) - frame_length, frame_length):
            frame = audio_16bit[i:i + frame_length]

            # VAD requires specific sample rates
            if sr != 16000:
                frame_resampled = librosa.resample(
                    frame.astype(np.float32), orig_sr=sr, target_sr=16000
                ).astype(np.int16)
            else:
                frame_resampled = frame

            # Check if frame contains speech
            is_speech = self.vad.is_speech(frame_resampled.tobytes(), 16000)
            voiced_frames.append(is_speech)

            if is_speech:
                start_time = i / sr
                end_time = (i + frame_length) / sr
                speech_segments.append((start_time, end_time))

        # Merge nearby speech segments
        if speech_segments:
            merged_segments = [speech_segments[0]]
            for start, end in speech_segments[1:]:
                if start - merged_segments[-1][1] < 0.5:  # Merge if gap < 0.5s
                    merged_segments[-1] = (merged_segments[-1][0], end)
                else:
                    merged_segments.append((start, end))

            speech_segments = merged_segments

        return audio, speech_segments

    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int, List[Tuple[float, float]]]:
        """Complete audio preprocessing pipeline"""
        logger.info(f"Loading audio: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        logger.info(f"Original: {len(audio)/sr:.2f}s, {sr}Hz")

        # Resample to target sample rate
        if sr != self.target_sr:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # Normalize
        audio = self.normalize_audio(audio)

        # Enhance speech (for noisy/distorted audio)
        # audio = self.enhance_speech(audio, sr)

        # Apply noise reduction
        audio = self.reduce_noise(audio, sr)

        # Apply VAD
        audio, speech_segments = self.apply_vad(audio, sr)

        logger.info(
            f"Preprocessing complete. Found {len(speech_segments)} speech segments")

        return audio, sr, speech_segments


class ProfessionalTranscriber:
    """Professional-grade transcription system using Faster-Whisper"""

    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor()

        # Load model
        logger.info(f"Loading Faster-Whisper model: {model_size}")
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"selected device:{device}")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=8,
            num_workers=4
        )
        logger.info("Model loaded successfully")

    def transcribe_audio(self, audio_path: str, language: str = "en",
                        #  vad_filter: bool = True, vad_parameters: dict = None) -> Dict[str, Any]:
                         vad_filter: bool = False, vad_parameters: dict = None) -> Dict[str, Any]:
        """Transcribe audio with optimal settings"""

        # Default VAD parameters optimized for your use case
        if vad_parameters is None:
            vad_parameters = {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "max_speech_duration_s": 30,
                "min_silence_duration_ms": 100,

                "speech_pad_ms": 400
            }

        # Preprocess audio
        audio, sr, speech_segments = self.preprocessor.preprocess_audio(
            audio_path)

        audio = audio.astype(np.float32)

        logger.info("Starting transcription...")
        start_time = time.time()

        # Transcribe with optimal parameters
        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=5,  # Good balance of accuracy and speed
            best_of=5,    # Multiple candidates for better accuracy
            temperature=0.0,  # Deterministic output
            condition_on_previous_text=True,  # Better context awareness
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            word_timestamps=True,  # For better SRT timing
            initial_prompt="This is a professional transcription. Please be accurate with technical terms, proper nouns, and punctuation."
        )

        # Process results
        transcription_segments = []
        full_text = []

        for segment in segments:
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "confidence": getattr(segment, 'avg_logprob', 0.0),
                "words": []
            }

            # Add word-level timestamps if available
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    word_dict = {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "confidence": word.probability
                    }
                    segment_dict["words"].append(word_dict)

            transcription_segments.append(segment_dict)
            full_text.append(segment.text.strip())

        end_time = time.time()

        # Compile results
        results = {
            "segments": transcription_segments,
            "full_text": " ".join(full_text),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "transcription_time": end_time - start_time,
            "speech_segments": speech_segments
        }

        logger.info(f"Transcription completed in {end_time - start_time:.2f}s")
        logger.info(
            f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

        return results



# // single spcae between words




    def generate_srt(self, results: Dict[str, Any], output_path: str = None, max_words_per_line: int = 5) -> str:
        """Generate SRT subtitle file with a maximum number of words per line, using word-level timestamps if available.
        Ensures only single spaces between words in each line.
        """
        import re
        if output_path is None:
            output_path = "transcription.srt"

        srt_content = []
        counter = 1

        for segment in results["segments"]:
            words = segment.get("words", [])
            # If we have word-level timestamps, use them to split and timestamp lines
            if words and len(words) > 1:
                line = []
                for idx, word in enumerate(words):
                    # Ensure word has no leading/trailing spaces
                    w = word.copy()
                    w['word'] = w['word'].strip()
                    line.append(w)
                    # If line full or last word
                    if len(line) == max_words_per_line or idx == len(words) - 1:
                        start_time = line[0]['start'] if 'start' in line[0] else segment['start']
                        end_time = line[-1]['end'] if 'end' in line[-1] else segment['end']
                        # Join with single space and collapse any double spaces
                        text = " ".join(w['word'] for w in line)
                        text = re.sub(r'\s+', ' ', text)  # Ensure only single spaces
                        srt_content.append(f"{counter}")
                        srt_content.append(
                            f"{self.seconds_to_srt_time(start_time)} --> {self.seconds_to_srt_time(end_time)}")
                        srt_content.append(text)
                        srt_content.append("")  # Empty line between segments
                        counter += 1
                        line = []
            else:
                # Fallback: Split by text if no word timestamps available
                text_words = [w.strip() for w in segment["text"].split()]
                start_time = segment["start"]
                end_time = segment["end"]
                total_words = len(text_words)
                duration = end_time - start_time

                for i in range(0, total_words, max_words_per_line):
                    line_words = text_words[i:i+max_words_per_line]
                    # Calculate proportional times
                    line_start = start_time + (i / total_words) * duration
                    line_end = start_time + \
                        (min(i+max_words_per_line, total_words) / total_words) * duration
                    text = " ".join(line_words)
                    text = re.sub(r'\s+', ' ', text)  # Ensure only single spaces
                    srt_content.append(f"{counter}")
                    srt_content.append(
                        f"{self.seconds_to_srt_time(line_start)} --> {self.seconds_to_srt_time(line_end)}")
                    srt_content.append(text)
                    srt_content.append("")
                    counter += 1

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(srt_content))

        logger.info(f"SRT file saved: {output_path}")
        return output_path




    def seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    



# ... [rest of your imports and class definitions stay the same] ...

def transcribe_files(audio_file_paths: List[str], output_dir: str = ".", model_size: str = "large-v3"):
    """
    Transcribe a list of audio files and save their SRT files.
    """
    # Initialize transcriber once for efficiency
    transcriber = ProfessionalTranscriber(model_size=model_size)
    summary = []

    for audio_file_path in audio_file_paths:
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            continue

        base_name = Path(audio_file_path).stem
        srt_path = os.path.join(output_dir, f"{base_name}.srt")

        # Transcribe
        results = transcriber.transcribe_audio(
            audio_path=audio_file_path,
            language="en",
            vad_filter=True
        )
        transcriber.generate_srt(results, srt_path)
        logger.info(f"Transcription complete for {audio_file_path}, SRT saved to {srt_path}")

        summary.append({
            "file": audio_file_path,
            "srt": srt_path,
            "duration": results['duration'],
            "language": results['language'],
            "probability": results['language_probability'],
            "transcription_time": results['transcription_time'],
            "length": len(results['full_text'])
        })
    return summary

if __name__ == "__main__":
    import glob

    # Default settings
    AUDIO_FILE_PATHS = [r"D:\office work audios\1.mp3"]  # List of files by default
    OUTPUT_DIR = "./transcriptions"
    MODEL_SIZE = "large-v3"

    # Command-line usage:
    # python script.py file1.wav file2.mp3 ... [or a directory]
    if len(sys.argv) > 1:
        # If argument is a directory, process all wav/mp3 in that folder
        arg = sys.argv[1]
        if os.path.isdir(arg):
            AUDIO_FILE_PATHS = glob.glob(os.path.join(arg, "*.wav")) + glob.glob(os.path.join(arg, "*.mp3"))
        else:
            AUDIO_FILE_PATHS = sys.argv[1:]  # List of files

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = transcribe_files(AUDIO_FILE_PATHS, OUTPUT_DIR, MODEL_SIZE)

    print("\n\n=== Batch Transcription Summary ===")
    for item in summary:
        print(f"File: {item['file']}")
        print(f"  SRT: {item['srt']}")
        print(f"  Duration: {item['duration']:.2f} sec")
        print(f"  Language: {item['language']} ({item['probability']:.2f})")
        print(f"  Transcript Length: {item['length']} chars")
        print(f"  Time Taken: {item['transcription_time']:.2f} sec\n")
    print(f"✅ {len(summary)} files processed. SRTs saved in {OUTPUT_DIR}")