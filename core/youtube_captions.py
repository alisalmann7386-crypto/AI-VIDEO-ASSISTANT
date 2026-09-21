"""Retrieve accessible YouTube captions, preserving language and timestamps.

No method here guarantees access when YouTube blocks a hosting IP.
"""
from __future__ import annotations

from html import unescape
import math
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    AgeRestricted, IpBlocked, NoTranscriptFound, RequestBlocked,
    TranscriptsDisabled, VideoUnavailable, YouTubeTranscriptApi,
)

from utils.audio_processor import MAX_DURATION_SECONDS, is_youtube_url

MAX_TRANSCRIPT_CHARS = 120_000


class CaptionsUnavailable(RuntimeError):
    """No accessible transcript, or YouTube denied access."""


def extract_video_id(url: str) -> str:
    """Extract a video ID after validating the URL's hostname."""
    if not is_youtube_url(url):
        raise ValueError("Enter a valid YouTube video URL.")
    parsed = urlparse(url.strip())
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/")
    if parsed.path == "/watch":
        return parse_qs(parsed.query)["v"][0]
    return parsed.path.strip("/").split("/")[-1]


def select_caption_track(tracks, language: str):
    """Prefer the user's language, then another available original-language track.

    A missing English/Hindi track is not the same as a video having no captions.
    The fallback does not translate captions and displays the actual track language.
    """
    preferred = (["hi", "hi-Latn", "en", "en-IN", "en-US", "en-GB"]
                 if language.lower() == "hinglish"
                 else ["en", "en-US", "en-GB", "en-IN", "hi", "hi-Latn"])
    try:
        return tracks.find_transcript(preferred)
    except NoTranscriptFound:
        return next(iter(tracks), None)


def fetch_youtube_captions(url: str, language: str = "english") -> tuple[dict, dict]:
    """Return a transcription-like dict and metadata for accessible YouTube captions."""
    video_id = extract_video_id(url)
    if language.lower() not in {"english", "hinglish"}:
        raise ValueError("Choose English or Hinglish.")
    try:
        tracks = YouTubeTranscriptApi().list(video_id)
        chosen = select_caption_track(tracks, language)
        if chosen is None:
            raise CaptionsUnavailable("This video has no available captions.")
        fetched = chosen.fetch()
    except (IpBlocked, RequestBlocked) as exc:
        raise CaptionsUnavailable(
            "YouTube has blocked caption requests from this server's IP address. "
            "Use Paste transcript or Upload a file; retrying the same server may not help."
        ) from exc
    except TranscriptsDisabled as exc:
        raise CaptionsUnavailable("Captions are disabled for this video.") from exc
    except (VideoUnavailable, AgeRestricted) as exc:
        raise CaptionsUnavailable("The video is unavailable or requires authorization to access captions.") from exc
    except CaptionsUnavailable:
        raise
    except Exception as exc:
        # Do not surface upstream URLs or other sensitive request details in the UI.
        raise CaptionsUnavailable("Could not retrieve YouTube captions from this server.") from exc

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
    track_language = str(getattr(chosen, "language", "Unknown language"))
    transcript = {"full_text": " ".join(parts), "segments": segments}
    seconds = int(last_end)
    metadata = {
        "title": "YouTube video (captions)", "channel": "YouTube captions",
        "duration": f"{seconds // 60}m {seconds % 60}s",
        "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "url": url.strip(), "transcript_source": f"YouTube captions ({track_language})",
    }
    return transcript, metadata
