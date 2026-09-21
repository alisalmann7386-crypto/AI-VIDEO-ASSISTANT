"""YouTube caption regression tests; no actual YouTube or paid API calls."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.youtube_captions import (
    CaptionsUnavailable, extract_video_id, fetch_youtube_captions,
)

VIDEO = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class CaptionsTests(unittest.TestCase):
    def test_extract_id_for_supported_video_urls(self):
        for url in (
            VIDEO, "https://youtu.be/dQw4w9WgXcQ?t=4",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/live/dQw4w9WgXcQ",
        ):
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")
        with self.assertRaises(ValueError):
            extract_video_id("https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ")

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_available_caption_segments_with_timestamps(self, api_class):
        api_class.return_value.fetch.return_value = [
            SimpleNamespace(text="Hello &amp; welcome", start=0, duration=2.5),
            SimpleNamespace(text="To the meeting", start=2.5, duration=2),
        ]
        result, metadata = fetch_youtube_captions(VIDEO)
        api_class.return_value.fetch.assert_called_once_with("dQw4w9WgXcQ", languages=["en", "en-US", "en-GB"])
        self.assertEqual(result["full_text"], "Hello & welcome To the meeting")
        self.assertEqual(result["segments"][1]["start"], "00:02")
        self.assertEqual(result["segments"][1]["end"], "00:04")
        self.assertEqual(metadata["transcript_source"], "YouTube captions")
        self.assertEqual(metadata["url"], VIDEO)

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_hinglish_prefers_hindi_then_english(self, api_class):
        api_class.return_value.fetch.return_value = [SimpleNamespace(text="Namaste", start=0, duration=1)]
        result, _ = fetch_youtube_captions(VIDEO, language="hinglish")
        self.assertEqual(result["full_text"], "Namaste")
        api_class.return_value.fetch.assert_called_once_with("dQw4w9WgXcQ", languages=["hi", "hi-Latn", "en", "en-IN"])

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_blocked_caption_requests_get_safe_message(self, api_class):
        api_class.return_value.fetch.side_effect = Exception("upstream error with private URL")
        with self.assertRaisesRegex(CaptionsUnavailable, "unavailable from this server") as error:
            fetch_youtube_captions(VIDEO)
        self.assertNotIn("private URL", str(error.exception))

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_empty_captions_are_unavailable(self, api_class):
        api_class.return_value.fetch.return_value = [SimpleNamespace(text="   ", start=0, duration=1)]
        with self.assertRaises(CaptionsUnavailable):
            fetch_youtube_captions(VIDEO)

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_long_video_is_rejected(self, api_class):
        api_class.return_value.fetch.return_value = [SimpleNamespace(text="Late caption", start=7200, duration=10)]
        with self.assertRaisesRegex(CaptionsUnavailable, "two-hour"):
            fetch_youtube_captions(VIDEO)


if __name__ == "__main__":
    unittest.main()
