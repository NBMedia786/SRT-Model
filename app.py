import streamlit as st
import os
from pathlib import Path
import json
import subprocess
from transcription_system import ProfessionalTranscriber
import urllib.parse

# Paths
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
SUMMARY_PATH = DATA_DIR / "summary.json"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.touch(exist_ok=True)

# Load summary
try:
    with open(SUMMARY_PATH, 'r') as f:
        summary_data = json.load(f)
except Exception:
    summary_data = {}

# Transcriber instance
transcriber = ProfessionalTranscriber()

# ---- Sidebar Upload Panel ----
st.sidebar.title("🎙️ Upload Audio/Video")
uploaded_file = st.sidebar.file_uploader("Upload your file", type=["mp3", "wav", "mp4", "m4a"])

def convert_to_wav(input_path: Path, output_path: Path):
    """Converts audio/video file to WAV format (mono, 16kHz)."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",  # overwrite if exists
                "-i", str(input_path),
                "-ac", "1",      # mono channel
                "-ar", "16000",  # 16 kHz
                str(output_path)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        st.error("❌ Audio conversion failed. Please ensure ffmpeg is installed.")
        raise


if uploaded_file:
    original_path = AUDIO_DIR / uploaded_file.name
    with open(original_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

    # Convert to .wav
    wav_path = AUDIO_DIR / f"{original_path.stem}.wav"
    if original_path.suffix.lower() != ".wav":
        with st.spinner("Converting to WAV..."):
            convert_to_wav(original_path, wav_path)
    else:
        wav_path = original_path

    # Transcription
    with st.spinner("Transcribing..."):
        result = transcriber.transcribe_audio(str(wav_path))
        srt_path = TRANSCRIPT_DIR / f"{wav_path.stem}.srt"
        transcriber.generate_speaker_attributed_srt(str(wav_path), result, str(srt_path))

    summary_data[wav_path.name] = {
        "audio_path": str(wav_path),
        "srt_path": str(srt_path),
        "full_text": result['full_text'],
        "language": result['language'],
        "probability": result['language_probability'],
        "transcription_time": result['transcription_time']
    }
    with open(SUMMARY_PATH, 'w') as f:
        json.dump(summary_data, f, indent=2)
    st.sidebar.success("✅ Transcription complete!")


# ---- Routing: Main Grid View vs Detail View ----
query_params = st.experimental_get_query_params()
selected_file = query_params.get("file", [None])[0]

if selected_file:
    # ---- Detail View ----
    st.title(f"📄 Transcript: {selected_file}")
    data = summary_data.get(selected_file)
    if not data:
        st.error("File not found.")
    else:
        st.subheader("📝 Full Transcript")
        st.write(data["full_text"])

        col1, col2 = st.columns(2)
        with col1:
            with open(data["srt_path"], "rb") as f:
                st.download_button("📥 Download SRT", f, file_name=Path(data["srt_path"]).name, mime="text/plain")
        with col2:
            if st.button("♻️ Regenerate Transcript"):
                with st.spinner("Re-transcribing..."):
                    result = transcriber.transcribe_audio(data["audio_path"])
                    srt_path = TRANSCRIPT_DIR / f"{Path(data['audio_path']).stem}.srt"
                    # transcriber.generate_srt(result, str(srt_path))
                    transcriber.generate_speaker_attributed_srt(str(file_path), result, str(srt_path))
                    summary_data[selected_file] = {
                        "audio_path": data["audio_path"],
                        "srt_path": str(srt_path),
                        "full_text": result['full_text'],
                        "language": result['language'],
                        "probability": result['language_probability'],
                        "transcription_time": result['transcription_time']
                    }
                    with open(SUMMARY_PATH, 'w') as f:
                        json.dump(summary_data, f, indent=2)
                st.success("🔁 Transcript regenerated!")
                st.rerun()

else:
    # ---- Main Grid View ----
    st.title("🗂️ Transcribed Files")

    if not summary_data:
        st.info("No transcriptions yet. Upload a file to get started!")
    else:
        # Show files in a responsive grid
        col_count = 5
        keys = list(summary_data.keys())
        rows = [keys[i:i+col_count] for i in range(0, len(keys), col_count)]

        for row in rows:
            cols = st.columns(len(row))
            for i, filename in enumerate(row):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #ddd; border-radius:12px; padding:10px; text-align:center; height:100px; display:flex; align-items:center; justify-content:center;">
                            <a href="/?file={urllib.parse.quote(filename)}" target="_blank" style="text-decoration:none; color:inherit;">
                                <strong>{filename}</strong>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
