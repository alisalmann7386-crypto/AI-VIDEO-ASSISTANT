"""Fast regression tests; no external APIs, tokens or media downloads required."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from utils.audio_processor import is_youtube_url, process_local_file
from core.transcriber import format_timestamp, transcribe_all
from core.rag_engine import cosine_similarity, prepare_documents
from core.summarizer import split_transcript


class VideoAssistantTests(unittest.TestCase):
    def test_youtube_url_allowlist(self):
        self.assertTrue(is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_youtube_url("https://youtu.be/dQw4w9WgXcQ"))
        self.assertTrue(is_youtube_url("https://youtube.com/shorts/dQw4w9WgXcQ"))
        self.assertFalse(is_youtube_url("https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ"))
        self.assertFalse(is_youtube_url("https://example.com/video.mp4"))
        self.assertFalse(is_youtube_url("file:///etc/passwd"))

    def test_timestamp(self):
        self.assertEqual(format_timestamp(65), "01:05")
        self.assertEqual(format_timestamp(3661), "01:01:01")

    def test_document_preparation(self):
        docs = prepare_documents([{"start": "00:12", "end": "00:18", "text": "important discovery"}])
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].start, "00:12")
        self.assertEqual(docs[0].text, "important discovery")

    def test_cosine(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0)
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0)
        self.assertEqual(cosine_similarity([1], [1, 2]), 0)

    def test_split_long_transcript(self):
        chunks = split_transcript("hello world " * 120, chunk_size=150)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(part) <= 150 for part in chunks))
        self.assertEqual(" ".join(chunks), ("hello world " * 120).strip())

    def test_transcriber_offsets_and_missing_key(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "part.mp3"
            audio.write_bytes(b"test")
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
                with patch("core.transcriber.transcribe_chunk_groq", return_value=("hello", [{"start": "00:00", "end": "00:04", "text": "hello"}])):
                    with patch("core.transcriber.get_audio_duration", return_value=4.0):
                        result = transcribe_all([str(audio)])
            self.assertEqual(result["full_text"], "hello")
            self.assertEqual(len(result["segments"]), 1)

    def test_ffmpeg_media_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "sample.wav"
            try:
                subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-y", str(source)], check=True, capture_output=True, timeout=30)
            except FileNotFoundError:
                self.skipTest("ffmpeg is not installed in this test environment")
            chunks, metadata = process_local_file(str(source))
            self.assertTrue(chunks)
            self.assertTrue(all(Path(chunk).is_file() for chunk in chunks))
            self.assertEqual(metadata["channel"], "Local upload")


if __name__ == "__main__":
    unittest.main()
