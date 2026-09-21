"""Groq-hosted Whisper transcription with global timestamps and no local ML weights."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from groq import Groq
from utils.audio_processor import get_audio_duration


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _field(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def transcribe_chunk_groq(chunk_path: str, api_key: str, time_offset: float = 0.0) -> tuple[str, list[dict]]:
    """Transcribe with multilingual Whisper; return readable global-time segments."""
    client = Groq(api_key=api_key)
    with open(chunk_path, "rb") as audio:
        response = client.audio.transcriptions.create(
            file=(Path(chunk_path).name, audio),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    text = str(_field(response, "text", "") or "").strip()
    raw_segments = _field(response, "segments", []) or []
    segments: list[dict] = []
    for segment in raw_segments:
        content = str(_field(segment, "text", "") or "").strip()
        if not content:
            continue
        start = float(_field(segment, "start", 0.0) or 0.0) + time_offset
        end = float(_field(segment, "end", 0.0) or 0.0) + time_offset
        segments.append({"start": format_timestamp(start), "end": format_timestamp(end),
                         "start_raw": start, "end_raw": end, "text": content})
    if text and not segments:
        end = time_offset + get_audio_duration(chunk_path)
        segments = [{"start": format_timestamp(time_offset), "end": format_timestamp(end),
                     "start_raw": time_offset, "end_raw": end, "text": text}]
    return text, segments


def transcribe_all(chunks: list[str], language: str = "english") -> dict:
    """Process English or Hinglish audio; Whisper auto-detects spoken language.

    Hinglish is transcribed, not translated. No Sarvam subscription is needed.
    """
    if language.lower() not in {"english", "hinglish"}:
        raise ValueError("Choose English or Hinglish.")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Configure it in Streamlit app secrets.")
    if not chunks:
        raise ValueError("No audio chunks were generated.")
    full_text: list[str] = []
    all_segments: list[dict] = []
    offset = 0.0
    for chunk in chunks:
        if not Path(chunk).is_file():
            raise ValueError("A required audio chunk is missing.")
        text, segments = transcribe_chunk_groq(chunk, api_key, offset)
        if text:
            full_text.append(text)
            all_segments.extend(segments)
        offset += get_audio_duration(chunk)
    if not full_text:
        raise ValueError("No speech was detected. Try a video with clear spoken audio.")
    return {"full_text": " ".join(full_text), "segments": all_segments}
