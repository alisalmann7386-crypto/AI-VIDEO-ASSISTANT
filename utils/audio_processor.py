"""Cloud-friendly media extraction. Requires ffmpeg and ffprobe (packages.txt)."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import parse_qs, urlparse

import yt_dlp

MAX_DURATION_SECONDS = 2 * 60 * 60
ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com", "www.youtube-nocookie.com", "youtu.be", "www.youtu.be"}


def is_youtube_url(value: str) -> bool:
    """Accept only well-formed YouTube video, Shorts and short URLs."""
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ALLOWED_HOSTS:
        return False
    if parsed.username or parsed.password or parsed.port:
        return False
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return bool(re.fullmatch(r"/[A-Za-z0-9_-]{11}/?", parsed.path))
    if parsed.path == "/watch":
        return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", parse_qs(parsed.query).get("v", [""])[0]))
    return bool(re.fullmatch(r"/(shorts|live|embed)/[A-Za-z0-9_-]{11}/?", parsed.path))


def _run(command: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg/ffprobe is missing. Add ffmpeg to packages.txt.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Media conversion timed out; try a shorter recording.") from exc
    except subprocess.CalledProcessError as exc:
        # Never echo arbitrary video metadata or command output in the public UI.
        raise RuntimeError("Could not decode this media. Try MP3, WAV, MP4, M4A or WEBM.") from exc


def get_audio_duration(path: str | Path) -> float:
    result = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], timeout=30)
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("Could not determine audio duration.") from exc


def split_audio_into_chunks(audio_path: str, chunk_length_ms: int = 10 * 60 * 1000) -> list[str]:
    """Normalize audio to low-bandwidth MP3 chunks, avoiding pydub/audioop.

    The 64-kbps mono audio chunks are below common transcription upload limits.
    """
    if chunk_length_ms <= 0:
        raise ValueError("Audio chunk length must be positive.")
    duration = get_audio_duration(audio_path)
    if duration <= 0:
        raise ValueError("The uploaded file contains no readable audio.")
    if duration > MAX_DURATION_SECONDS:
        raise ValueError("Audio longer than two hours is not supported; upload a shorter clip.")
    output_dir = Path(audio_path).parent / "audio_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "part_%03d.mp3")
    _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000",
        "-b:a", "64k", "-f", "segment", "-segment_time", str(chunk_length_ms / 1000),
        "-reset_timestamps", "1", output_pattern,
    ], timeout=600)
    chunks = sorted(str(path) for path in output_dir.glob("part_*.mp3"))
    if not chunks:
        raise RuntimeError("No audio could be extracted. Check that the file contains an audio track.")
    return chunks


def download_youtube_audio(url: str, work_dir: str | None = None) -> tuple[list[str], dict]:
    """Download publicly accessible video audio. Cloud IP restrictions may still apply."""
    if not is_youtube_url(url):
        raise ValueError("Enter a valid YouTube video URL (watch, Shorts or youtu.be).")
    directory = Path(work_dir or tempfile.mkdtemp(prefix="video_assistant_"))
    directory.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best", "outtmpl": str(directory / "%(id)s.%(ext)s"),
        "noplaylist": True, "quiet": True, "no_warnings": True,
        "socket_timeout": 25, "retries": 2, "fragment_retries": 2,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError("YouTube blocked this download or the video is unavailable. Try uploading an audio/video file instead; cloud IP restrictions cannot be guaranteed away.") from exc
    if not isinstance(info, dict):
        raise RuntimeError("YouTube did not return video information.")
    audio_files = list(directory.glob("*.mp3"))
    if not audio_files:
        raise RuntimeError("No audio file was produced from this YouTube video.")
    duration = int(info.get("duration") or 0)
    if duration > MAX_DURATION_SECONDS:
        raise ValueError("Videos longer than two hours are not supported.")
    metadata = {
        "title": str(info.get("title") or "YouTube Video"),
        "channel": str(info.get("uploader") or "Unknown channel"),
        "duration": f"{duration // 60}m {duration % 60}s" if duration else "Unknown",
        "thumbnail": info.get("thumbnail"), "url": url,
    }
    return split_audio_into_chunks(str(audio_files[0])), metadata


def process_local_file(file_path: str) -> tuple[list[str], dict]:
    """Process a file within its caller-owned temporary directory."""
    path = Path(file_path)
    if not path.is_file():
        raise ValueError("The uploaded media file could not be found.")
    duration = get_audio_duration(path)
    if duration <= 0:
        raise ValueError("The file has no readable audio track.")
    if duration > MAX_DURATION_SECONDS:
        raise ValueError("Videos longer than two hours are not supported.")
    metadata = {
        "title": path.stem, "channel": "Local upload",
        "duration": f"{int(duration) // 60}m {int(duration) % 60}s",
        "thumbnail": None, "url": None,
    }
    return split_audio_into_chunks(str(path)), metadata


def process_input(source: str, work_dir: str | None = None) -> tuple[list[str], dict]:
    """Route valid YouTube URLs or existing local media paths."""
    if is_youtube_url(source):
        return download_youtube_audio(source, work_dir)
    if source.strip().lower().startswith(("http://", "https://")):
        raise ValueError("Only YouTube video URLs are supported. Other videos must be uploaded as files.")
    return process_local_file(source)
