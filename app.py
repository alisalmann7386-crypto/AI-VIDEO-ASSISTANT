"""Streamlit UI for captions, uploads or pasted transcripts and grounded analysis."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import is_youtube_url, process_input
from core.youtube_captions import CaptionsUnavailable, MAX_TRANSCRIPT_CHARS, extract_video_id, fetch_youtube_captions
from core.transcriber import transcribe_all
from core.summarizer import generate_title, summarize
from core.extractor import extract_insights
from core.rag_engine import ask_question, build_rag_chain
from core.groq_analysis import analyze_with_groq, build_groq_rag, is_rate_limited

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


def has_key(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def safe_error(exc: Exception) -> str:
    """Keep provider credentials and verbose upstream API responses out of the UI."""
    if is_rate_limited(exc):
        return ("The selected AI provider has reached its API rate or usage limit (HTTP 429). "
                "Check its developer dashboard or use the other provider. Your transcript is saved "
                "in this browser session; retry analysis without extracting it again.")
    message = str(exc)
    for variable in ("GROQ_API_KEY", "MISTRAL_API_KEY"):
        secret = os.getenv(variable, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:500] or type(exc).__name__


def analyze_saved(data: dict, provider: str) -> None:
    """Analyze cached transcript; preserve the text even when a provider rejects a call."""
    transcript = data["full_transcript"]
    segments = data.get("segments", [])
    if provider == "Groq":
        result = analyze_with_groq(transcript)
        rag_chain = build_groq_rag(segments or transcript)
        data.update({**result, "rag_chain": rag_chain, "provider": "Groq"})
        return
    try:
        title = generate_title(transcript)
        summary = summarize(transcript)
        insights = extract_insights(summary)
        rag_chain = build_rag_chain(segments or transcript)
        data.update({"title": title, "summary": summary, "insights": insights,
                     "rag_chain": rag_chain, "coverage_note": "", "provider": "Mistral"})
    except Exception as exc:
        # Do not silently switch for invalid keys or other unexpected failures.
        if not is_rate_limited(exc) or not has_key("GROQ_API_KEY"):
            raise
        st.info("Mistral reached its API limit. Switching to Groq without reprocessing your video…")
        result = analyze_with_groq(transcript)
        data.update({**result, "rag_chain": build_groq_rag(segments or transcript),
                     "provider": "Groq (Mistral rate-limit fallback)"})


with st.sidebar:
    st.title("🎛️ Control panel")
    st.caption("Analyze public YouTube captions, video audio, or a transcript you provide.")
    source_type = st.radio("Input source", ["YouTube URL", "Upload a file", "Paste transcript"])
    url = ""
    uploaded = None
    pasted_text = ""
    reference_url = ""
    if source_type == "YouTube URL":
        url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("Captions first, then audio. YouTube can block both from cloud servers.")
    elif source_type == "Upload a file":
        uploaded = st.file_uploader("MP3, WAV, M4A, MP4, MKV or WEBM", type=["mp3", "wav", "m4a", "mp4", "mkv", "webm"])
    else:
        pasted_text = st.text_area("Video transcript", height=200, placeholder="Paste the video's transcript here…")
        reference_url = st.text_input("YouTube link (optional)", placeholder="https://www.youtube.com/watch?v=...")
        st.caption("Copy YouTube's Show transcript text here; plain text does not preserve timestamps.")
    language = (st.selectbox("Spoken language", ["english", "hinglish"],
                            help="Captions keep their original language; Whisper transcribes rather than translates.")
                if source_type != "Paste transcript" else "english")
    providers = (["Groq", "Mistral"] if has_key("GROQ_API_KEY") else ["Mistral", "Groq"])
    provider = st.selectbox("AI analysis provider", providers,
                           help="Groq uses one structured request for title, summary and insights and local transcript search. Mistral may use multiple requests and embeddings.")
    st.caption("Maximum media length: 2 hours; pasted text: 120,000 characters.")
    start = st.button("🚀 Analyze content", type="primary", use_container_width=True)
    if provider == "Groq" and not has_key("GROQ_API_KEY"):
        st.warning("Add GROQ_API_KEY to Streamlit Secrets for Groq analysis.")
    if provider == "Mistral" and not has_key("MISTRAL_API_KEY"):
        st.warning("Add MISTRAL_API_KEY to Streamlit Secrets for Mistral analysis.")
    if source_type == "Upload a file" and not has_key("GROQ_API_KEY"):
        st.warning("GROQ_API_KEY is required to transcribe uploaded media.")

st.title("🎬 AI Video Assistant")
st.caption("Captions or transcription → summary → key insights → transcript-grounded Q&A.")

if start:
    st.session_state.analysis_data = None
    st.session_state.chat_history = []
    if not has_key("GROQ_API_KEY" if provider == "Groq" else "MISTRAL_API_KEY"):
        st.error(f"Missing API key for {provider}. Configure it in Streamlit Secrets.")
    elif source_type == "YouTube URL" and not is_youtube_url(url):
        st.error("Paste a valid YouTube watch, Shorts or youtu.be link.")
    elif source_type == "Upload a file" and uploaded is None:
        st.error("Upload an audio or video file before starting.")
    elif source_type == "Upload a file" and not has_key("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is required for audio transcription.")
    elif source_type == "Paste transcript" and not pasted_text.strip():
        st.error("Paste the video transcript before starting.")
    elif source_type == "Paste transcript" and len(pasted_text) > MAX_TRANSCRIPT_CHARS:
        st.error("The transcript is too long. Use fewer than 120,000 characters.")
    elif source_type == "Paste transcript" and reference_url.strip() and not is_youtube_url(reference_url):
        st.error("The optional reference link must be a valid YouTube URL.")
    else:
        with st.status("Processing content…", expanded=True) as status:
            try:
                with tempfile.TemporaryDirectory(prefix="ai_video_") as workspace:
                    if source_type == "Paste transcript":
                        st.write("📝 1/4 · Using your pasted transcript (no YouTube download needed)…")
                        transcript_data = {"full_text": pasted_text.strip(), "segments": []}
                        metadata = {"title": "Pasted video transcript", "channel": "Transcript supplied by user",
                                    "duration": "Unknown", "thumbnail": None,
                                    "url": reference_url.strip() or None, "transcript_source": "Pasted transcript"}
                        if reference_url.strip():
                            video_id = extract_video_id(reference_url)
                            metadata["thumbnail"] = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
                    elif source_type == "YouTube URL":
                        st.write("📝 1/4 · Trying available YouTube captions…")
                        try:
                            transcript_data, metadata = fetch_youtube_captions(url.strip(), language=language)
                            st.write("✅ Captions found; no download or audio transcription required.")
                        except CaptionsUnavailable as caption_error:
                            st.write("Captions unavailable; attempting audio download…")
                            if not has_key("GROQ_API_KEY"):
                                raise RuntimeError(
                                    f"{caption_error} GROQ_API_KEY is needed for audio transcription. "
                                    "Otherwise select Paste transcript or Upload a file."
                                ) from None
                            try:
                                chunks, metadata = process_input(url.strip(), work_dir=workspace)
                            except RuntimeError as exc:
                                if "YouTube blocked this download" in str(exc) or "No audio file was produced" in str(exc):
                                    raise RuntimeError(
                                        f"{caption_error} Audio is unavailable too. "
                                        "Paste the transcript or upload media you may process."
                                    ) from exc
                                raise
                            st.write(f"🎙️ 2/4 · Transcribing {len(chunks)} audio chunk(s)…")
                            transcript_data = transcribe_all(chunks, language=language)
                            metadata["transcript_source"] = "Groq Whisper audio transcription"
                    else:
                        suffix = Path(uploaded.name).suffix.lower()
                        file_path = Path(workspace) / f"upload{suffix}"
                        uploaded.seek(0)
                        with file_path.open("wb") as destination:
                            while block := uploaded.read(1024 * 1024):
                                destination.write(block)
                        st.write("📥 1/4 · Extracting and normalizing uploaded audio…")
                        chunks, metadata = process_input(str(file_path), work_dir=workspace)
                        st.write(f"🎙️ 2/4 · Transcribing {len(chunks)} audio chunk(s)…")
                        transcript_data = transcribe_all(chunks, language=language)
                        metadata["transcript_source"] = "Groq Whisper audio transcription"
                    transcript = transcript_data["full_text"]
                    segments = transcript_data.get("segments", [])
                    if not transcript.strip():
                        raise ValueError("The supplied content contains no readable transcript.")
                    # Persist text BEFORE the first paid chat/embedding request.
                    st.session_state.analysis_data = {
                        "title": metadata.get("title") or "Video transcript", "metadata": metadata,
                        "full_transcript": transcript, "segments": segments,
                        "summary": None, "insights": {}, "rag_chain": None,
                        "coverage_note": "", "provider": provider,
                    }
                st.write("🧠 3/4 · Generating title, summary and insights…")
                analyze_saved(st.session_state.analysis_data, provider)
                st.write("🔎 4/4 · Preparing transcript-grounded Q&A…")
                status.update(label="✅ Analysis complete", state="complete", expanded=False)
            except Exception as exc:
                if st.session_state.analysis_data is not None:
                    st.session_state.analysis_data["analysis_error"] = safe_error(exc)
                    status.update(label="⚠️ Transcript saved; AI analysis incomplete", state="error", expanded=True)
                    st.warning(safe_error(exc))
                    st.info("The transcript below is retained. Retry analysis after checking your provider's API limits; no re-upload or YouTube download is needed.")
                else:
                    status.update(label="❌ Content retrieval failed", state="error", expanded=True)
                    st.error(safe_error(exc))
                    st.info("For YouTube restrictions, try Paste transcript or Upload a file.")

# Retry *only* the AI stage from the transcript saved in Streamlit session state.
if st.session_state.analysis_data is not None and st.session_state.analysis_data.get("summary") is None:
    st.warning("Your transcript is saved in this session, but the AI provider did not finish the analysis.")
    if st.button("🔄 Retry analysis from saved transcript", use_container_width=True):
        try:
            with st.spinner("Analyzing your saved transcript (no media download)…"):
                analyze_saved(st.session_state.analysis_data, provider)
                st.session_state.analysis_data.pop("analysis_error", None)
            st.success("Analysis complete.")
        except Exception as exc:
            st.warning(safe_error(exc))

if st.session_state.analysis_data is None:
    st.info("👈 Choose a YouTube link, upload media, or paste a transcript in the sidebar.")
else:
    data = st.session_state.analysis_data
    metadata = data["metadata"]
    image_col, info_col = st.columns([1, 2])
    with image_col:
        if metadata.get("thumbnail"):
            st.image(metadata["thumbnail"], use_container_width=True)
        else:
            st.info("📝 Transcript / uploaded media")
    with info_col:
        st.subheader(data["title"])
        st.write("**Source:**", metadata.get("channel") or "Local upload")
        st.write("**Transcript obtained via:**", metadata.get("transcript_source", "Audio transcription"))
        st.write("**Duration:**", metadata.get("duration") or "Unknown")
        if data.get("summary"):
            st.caption("AI analysis by " + data.get("provider", "the configured provider"))
        if metadata.get("url"):
            st.link_button("▶ Watch on YouTube", metadata["url"])
    if data.get("coverage_note"):
        st.warning(data["coverage_note"])
    summary_tab, insights_tab, transcript_tab, chat_tab = st.tabs([
        "📋 Summary", "💡 Insights", "📝 Transcript", "💬 Ask the video",
    ])
    with summary_tab:
        if data.get("summary"):
            st.markdown(data["summary"])
            st.download_button("⬇️ Download summary", data["summary"], file_name="video_summary.txt", mime="text/plain")
        else:
            st.info("The transcript is saved. Retry analysis after the provider's rate limit resets or choose another provider.")
    with insights_tab:
        if not data.get("summary"):
            st.info("Insights will appear when analysis succeeds.")
        else:
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
        if data.get("rag_chain") is None:
            st.info("Complete the AI analysis to activate transcript Q&A.")
        else:
            st.caption("Answers use retrieved transcript excerpts, with timestamps when available. Groq retrieval uses local keyword matching.")
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
                        answer = "Couldn't answer that question: " + safe_error(exc)
                    st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
