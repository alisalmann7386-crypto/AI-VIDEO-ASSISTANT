"""Offline caption regression tests; never contact YouTube or paid APIs."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from youtube_transcript_api import IpBlocked, NoTranscriptFound, TranscriptsDisabled
from core.youtube_captions import CaptionsUnavailable, extract_video_id, fetch_youtube_captions

VIDEO_ID = "dQw4w9WgXcQ"
VIDEO = f"https://www.youtube.com/watch?v={VIDEO_ID}"


def track(language="English", text="Hello &amp; welcome"):
    return SimpleNamespace(
        language=language,
        language_code="en" if language == "English" else "es",
        fetch=lambda: [SimpleNamespace(text=text, start=2.5, duration=2)],
    )


class CaptionsTests(unittest.TestCase):
    def test_extract_id_for_supported_urls(self):
        for url in (VIDEO, f"https://youtu.be/{VIDEO_ID}?t=4", f"https://youtube.com/shorts/{VIDEO_ID}", f"https://www.youtube.com/live/{VIDEO_ID}"):
            with self.subTest(url=url):
                self.assertEqual(extract_video_id(url), VIDEO_ID)
        with self.assertRaises(ValueError):
            extract_video_id(f"https://youtube.com.evil.test/watch?v={VIDEO_ID}")

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_preferred_caption_with_timestamps(self, api_class):
        english = track()
        listing = api_class.return_value.list.return_value
        listing.find_transcript.return_value = english
        result, metadata = fetch_youtube_captions(VIDEO)
        api_class.return_value.list.assert_called_once_with(VIDEO_ID)
        self.assertIn("en", listing.find_transcript.call_args.args[0])
        self.assertEqual(result["full_text"], "Hello & welcome")
        self.assertEqual(result["segments"][0]["start"], "00:02")
        self.assertEqual(result["segments"][0]["end"], "00:04")
        self.assertEqual(metadata["transcript_source"], "YouTube captions (English)")

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_hinglish_prefers_hindi(self, api_class):
        listing = api_class.return_value.list.return_value
        listing.find_transcript.return_value = track("Hindi", "Namaste")
        result, metadata = fetch_youtube_captions(VIDEO, language="hinglish")
        self.assertEqual(listing.find_transcript.call_args.args[0][0], "hi")
        self.assertEqual(result["full_text"], "Namaste")
        self.assertEqual(metadata["transcript_source"], "YouTube captions (Hindi)")

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_uses_available_spanish_when_english_track_missing(self, api_class):
        listing = api_class.return_value.list.return_value
        listing.find_transcript.side_effect = NoTranscriptFound(VIDEO_ID, ["en"], listing)
        listing.__iter__.return_value = iter([track("Spanish", "Hola")])
        result, metadata = fetch_youtube_captions(VIDEO)
        self.assertEqual(result["full_text"], "Hola")
        self.assertEqual(metadata["transcript_source"], "YouTube captions (Spanish)")

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_ip_block_not_mislabeled_as_no_captions(self, api_class):
        api_class.return_value.list.side_effect = IpBlocked(VIDEO_ID)
        with self.assertRaisesRegex(CaptionsUnavailable, "blocked caption requests"):
            fetch_youtube_captions(VIDEO)

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_disabled_captions_have_specific_error(self, api_class):
        api_class.return_value.list.side_effect = TranscriptsDisabled(VIDEO_ID)
        with self.assertRaisesRegex(CaptionsUnavailable, "disabled"):
            fetch_youtube_captions(VIDEO)

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_upstream_error_does_not_leak_private_url(self, api_class):
        api_class.return_value.list.side_effect = Exception("private URL or secret")
        with self.assertRaises(CaptionsUnavailable) as captured:
            fetch_youtube_captions(VIDEO)
        self.assertNotIn("private URL", str(captured.exception))

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_no_tracks_and_empty_transcript(self, api_class):
        listing = api_class.return_value.list.return_value
        listing.find_transcript.side_effect = NoTranscriptFound(VIDEO_ID, ["en"], listing)
        listing.__iter__.return_value = iter([])
        with self.assertRaisesRegex(CaptionsUnavailable, "no available captions"):
            fetch_youtube_captions(VIDEO)
        listing.find_transcript.side_effect = None
        listing.find_transcript.return_value = track(text="   ")
        with self.assertRaisesRegex(CaptionsUnavailable, "usable captions"):
            fetch_youtube_captions(VIDEO)

    @patch("core.youtube_captions.YouTubeTranscriptApi")
    def test_caption_duration_limit(self, api_class):
        listing = api_class.return_value.list.return_value
        listing.find_transcript.return_value = SimpleNamespace(
            language="English",
            fetch=lambda: [SimpleNamespace(text="Late", start=7200, duration=10)],
        )
        with self.assertRaisesRegex(CaptionsUnavailable, "two-hour"):
            fetch_youtube_captions(VIDEO)


if __name__ == "__main__":
    unittest.main()
