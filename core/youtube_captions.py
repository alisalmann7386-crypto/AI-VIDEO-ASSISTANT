"""Best-effort YouTube caption retrieval without downloading a media stream.

This is NOT a way around YouTube's cloud-IP restrictions: captions can be blocked too.
"""
from __future__ import annotations

from html import unescape
import math
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from utils.audio_processor import MAX_DURATION_SECONDS, is_youtube_url

MAX_TRANSCRIPT_CHARS = 120_000


class CaptionsUnavailable(RuntimeError):
    """The video has no accessible captions, or YouTube denied access."""


def extract_video_id(url: str) -> str:
    """Extract an 11-character ID only after the URL allowlist validates the host."""
    if not is_youtube_url(url):
        raise ValueError("Enter a valid YouTube video URL.")
    parsed = urlparse(url.strip())
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/")
    if parsed.path == "/watch":
        return parse_qs(parsed.query)["v"][0]
    return parsed.path.strip("/").split("/")[-1]


def fetch_youtube_captions(url: str, language: str = "english") -> tuple[dict, dict]:
    """Return (transcription-like dict, UI metadata) for accessible captions.

    Prefers native English captions, or Hindi/English for the Hinglish setting.
    Does not silently translate or claim Whisper transcribed the captions.
    """
    video_id = extract_video_id(url)
    languages = ["hi", "hi-Latn", "en", "en-IN"] if language.lower() == "hinglish" else ["en", "en-US", "en-GB"]
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    except Exception as exc:
        # The library may raise TranscriptsDisabled, VideoUnavailable, IpBlocked,
        # RequestBlocked or an HTTP transport error. Never surface verbose upstream
        # messages containing request URLs or other identifiers in a public app.
        raise CaptionsUnavailable("YouTube captions are unavailable from this server.") from exc

    segments: list[dict] = []
    parts: list[str] = []
    total_chars = 0
    last_end = 0.0

    for snippet in fetched:
        text = " ".join(unescape(str(getattr(snippet, "text", ""))).split())
        if not text:
            continue
        try:
            start = float(getattr(snippet, "start", 0))
            duration = float(getattr(snippet, "duration", 0))
        except (TypeError, ValueError) as exc:
            raise CaptionsUnavailable("YouTube returned malformed caption timestamps.") from exc
        if not (math.isfinite(start) and math.isfinite(duration)) or start < 0 or duration < 0:
            raise CaptionsUnavailable("YouTube returned malformed caption timestamps.")
        end = start + duration
        if end > MAX_DURATION_SECONDS:
            raise CaptionsUnavailable("Captions exceed the two-hour limit; choose a shorter video.")
        total_chars += len(text) + 1
        if total_chars > MAX_TRANSCRIPT_CHARS:
            raise CaptionsUnavailable("Captions exceed the transcript size limit; use a shorter video.")
        from core.transcriber import format_timestamp
        segments.append({
            "start": format_timestamp(start), "end": format_timestamp(end),
            "start_raw": start, "end_raw": end, "text": text,
        })
        parts.append(text)
        last_end = max(last_end, end)

    if not parts:
        raise CaptionsUnavailable("This video did not provide usable captions.")
    transcript = {"full_text": " ".join(parts), "segments": segments}
    seconds = int(last_end)
    metadata = {
        "title": "YouTube video (captions)", "channel": "YouTube captions",
        "duration": f"{seconds // 60}m {seconds % 60}s",
        "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "url": url.strip(), "transcript_source": "YouTube captions",
    }
    return transcript, metadata
