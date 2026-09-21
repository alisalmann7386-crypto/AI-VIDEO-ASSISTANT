"""Streamlit entry point for the AI Video Assistant."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import is_youtube_url, process_input
from core.transcriber import transcribe_all
from core.summarizer import generate_title, summarize
from core.extractor import extract_insights
from core.rag_engine import ask_question, build_rag_chain

load_dotenv()
st.set_page_config(page_title="AI Video Assistant", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp {background: #0e1117; color: #e6edf3;}
section[data-testid="stSidebar"] {background: #161b22; border-right: 1px solid #30363d;}
.stButton > button[kind="primary"] {background: #d62929; color: white; border: 0;}
.stButton > button[kind="primary"]:hover {background: #b91c1c; color: white;}
[data-testid="stChatMessage"] {border: 1px solid #30363d; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def safe_error(exc: Exception) -> str:
    """Avoid displaying API credentials if an upstream SDK includes them in errors."""
    message = str(exc)
    for variable in ("GROQ_API_KEY", "MISTRAL_API_KEY"):
        secret = os.getenv(variable, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:500] or type(exc).__name__


def missing_keys() -> list[str]:
    return [key for key in ("GROQ_API_KEY", "MISTRAL_API_KEY") if not os.getenv(key, "").strip()]


with st.sidebar:
    st.title("🎛️ Control panel")
    st.caption("Analyze a public YouTube video or your own audio/video file.")
    source_type = st.radio("Input source", ["YouTube URL", "Upload a file"])
    url = ""
    uploaded = None
    if source_type == "YouTube URL":
        url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("Some public videos are blocked from cloud IP addresses. File upload is a reliable alternative.")
    else:
        uploaded = st.file_uploader("MP3, WAV, M4A, MP4, MKV or WEBM", type=["mp3", "wav", "m4a", "mp4", "mkv", "webm"])
    language = st.selectbox("Spoken language", ["english", "hinglish"], help="Whisper supports both languages. Hinglish transcription is not automatic translation.")
    st.caption("Maximum length: 2 hours. Large files may exceed Streamlit's upload limit.")
    start = st.button("🚀 Analyze media", type="primary", use_container_width=True)
    absent = missing_keys()
    if absent:
        st.warning("Add these keys in your app's Secrets settings: " + ", ".join(absent))

st.title("🎬 AI Video Assistant")
st.caption("Transcribe, summarize, extract insights and ask questions grounded in a video's transcript.")

if start:
    st.session_state.analysis_data = None
    st.session_state.chat_history = []
    if absent:
        st.error("Missing API credentials. Add the listed keys in Streamlit Secrets and restart the app.")
    elif source_type == "YouTube URL" and not is_youtube_url(url):
        st.error("Paste a valid YouTube watch, Shorts or youtu.be link.")
    elif source_type == "Upload a file" and uploaded is None:
        st.error("Upload an audio or video file before starting.")
    else:
        with st.status("Processing media…", expanded=True) as status:
            try:
                # Every run has a private disposable workspace; names from uploaded files
                # are not used as filesystem paths (prevents traversal and collisions).
                with tempfile.TemporaryDirectory(prefix="ai_video_") as workspace:
                    if uploaded is not None and source_type == "Upload a file":
                        suffix = Path(uploaded.name).suffix.lower()
                        file_path = Path(workspace) / f"upload{suffix}"
                        uploaded.seek(0)
                        with file_path.open("wb") as destination:
                            while block := uploaded.read(1024 * 1024):
                                destination.write(block)
                        source = str(file_path)
                    else:
                        source = url.strip()
                    st.write("📥 1/4 · Extracting and normalizing audio…")
                    chunks, metadata = process_input(source, work_dir=workspace)
                    st.write(f"🎙️ 2/4 · Transcribing {len(chunks)} audio chunk(s)…")
                    transcript_data = transcribe_all(chunks, language=language)
                    transcript = transcript_data["full_text"]
                    segments = transcript_data.get("segments", [])
                    if not transcript.strip():
                        raise ValueError("No intelligible speech was found in this file.")
                    st.write("🧠 3/4 · Summarizing and extracting key insights…")
                    title = generate_title(transcript)
                    summary = summarize(transcript)
                    insights = extract_insights(summary)
                    st.write("🔎 4/4 · Building a searchable transcript index…")
                    rag_chain = build_rag_chain(segments or transcript)
                    st.session_state.analysis_data = {
                        "title": title, "metadata": metadata,
                        "full_transcript": transcript, "segments": segments,
                        "summary": summary, "insights": insights, "rag_chain": rag_chain,
                    }
                status.update(label="✅ Analysis complete", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="❌ Analysis failed", state="error", expanded=True)
                st.error(f"{type(exc).__name__}: {safe_error(exc)}")
                st.info("For YouTube download errors, try a local file. For API errors, check your keys and provider quotas.")

if st.session_state.analysis_data is None:
    st.info("👈 Select a YouTube video or upload an audio/video file, then click **Analyze media**.")
else:
    data = st.session_state.analysis_data
    metadata = data["metadata"]
    image_col, info_col = st.columns([1, 2])
    with image_col:
        if metadata.get("thumbnail"):
            st.image(metadata["thumbnail"], use_container_width=True)
        else:
            st.info("🎧 Uploaded media")
    with info_col:
        st.subheader(data["title"])
        st.write("**Source:**", metadata.get("channel") or "Local upload")
        st.write("**Duration:**", metadata.get("duration") or "Unknown")
        if metadata.get("url"):
            st.link_button("▶ Watch on YouTube", metadata["url"])

    summary_tab, insights_tab, transcript_tab, chat_tab = st.tabs([
        "📋 Summary", "💡 Insights", "📝 Transcript", "💬 Ask the video",
    ])
    with summary_tab:
        st.markdown(data["summary"])
        st.download_button("⬇️ Download summary", data["summary"], file_name="video_summary.txt", mime="text/plain")
    with insights_tab:
        columns = st.columns(3)
        for column, heading, key in zip(columns, ["✅ Action items", "🔑 Key decisions", "❓ Open questions"],
                                        ["action_items", "key_decisions", "open_questions"]):
            with column:
                st.subheader(heading)
                for item in data["insights"].get(key, ["None discussed"]):
                    st.write("•", item)
    with transcript_tab:
        segments = data.get("segments", [])
        display = st.radio("View", ["Timestamped", "Full text"], horizontal=True) if segments else "Full text"
        if display == "Timestamped":
            for segment in segments:
                st.caption(f"⏱ {segment['start']} – {segment['end']}")
                st.write(segment["text"])
        else:
            st.text_area("Transcript", data["full_transcript"], height=400)
        st.download_button("⬇️ Download transcript", data["full_transcript"], file_name="transcript.txt", mime="text/plain")
    with chat_tab:
        st.caption("Answers are generated from retrieved transcript excerpts, with timestamps when available.")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        question = st.chat_input("Ask a question about this video…")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching transcript…"):
                        answer = ask_question(data["rag_chain"], question)
                except Exception as exc:
                    answer = f"Couldn't answer that question: {safe_error(exc)}"
                st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
