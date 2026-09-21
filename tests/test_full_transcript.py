"""Offline regression tests: full coverage, no real provider requests."""
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.groq_analysis import _synthesize, analyze_with_groq, split_all_transcript


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


FINAL = json.dumps({
    "title": "Whole video", "overview": "All topics were covered.",
    "key_takeaways": ["All parts"], "action_items": [],
    "key_decisions": [], "open_questions": [],
})


class FullTranscriptTests(unittest.TestCase):
    def test_lossless_split_preserves_every_character_and_order(self):
        transcript = "START αβ\n" + "word " * 4000 + "MIDDLE\n" + "z" * 9500 + " END"
        parts = split_all_transcript(transcript, chunk_size=8500)
        self.assertGreater(len(parts), 3)
        self.assertEqual("".join(parts), transcript)
        self.assertTrue(all(0 < len(chunk) <= 8500 for chunk in parts))
        self.assertIn("MIDDLE", "".join(parts))
        self.assertTrue(parts[-1].endswith("END"))

    @patch("core.groq_analysis.Groq")
    def test_long_transcript_summarizes_every_chunk_before_final(self, api):
        transcript = "START " + "alpha " * 3200 + "MIDDLE " + "beta " * 3200 + "END"
        parts = split_all_transcript(transcript)
        api.return_value.chat.completions.create.side_effect = [
            response(f"NOTES FOR PART {i + 1}") for i in range(len(parts))
        ] + [response(FINAL)]
        checkpoint = {}
        events = []
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy-key"}):
            output = analyze_with_groq(transcript, checkpoint=checkpoint, progress=events.append)
        calls = api.return_value.chat.completions.create.call_args_list
        self.assertEqual(len(calls), len(parts) + 1)
        for i, chunk in enumerate(parts):
            prompt = calls[i].kwargs["messages"][1]["content"]
            self.assertIn(chunk, prompt)
            self.assertIn(f"CHUNK {i + 1} OF {len(parts)}", prompt)
        final_prompt = calls[-1].kwargs["messages"][1]["content"]
        for i in range(len(parts)):
            self.assertIn(f"NOTES FOR PART {i + 1}", final_prompt)
        self.assertIn(f"All {len(parts)}", output["coverage_note"])
        self.assertNotIn("sampled passages", output["coverage_note"])
        self.assertEqual(len(checkpoint["notes"]), len(parts))
        self.assertTrue(any("chunk 1/" in msg for msg in events))

    @patch("core.groq_analysis.Groq")
    def test_failed_chunk_can_resume_without_repaying_completed_chunks(self, api):
        transcript = "INTRO " + "one " * 5000 + "FINAL " + "two " * 1500
        parts = split_all_transcript(transcript)
        self.assertGreaterEqual(len(parts), 3)
        checkpoint = {}
        api.return_value.chat.completions.create.side_effect = [
            response("COMPLETED PART 1"), RuntimeError("429 rate limit exceeded")
        ]
        with patch.dict(os.environ, {"GROQ_API_KEY": "dummy-key"}):
            with self.assertRaisesRegex(RuntimeError, "429"):
                analyze_with_groq(transcript, checkpoint=checkpoint)
            self.assertEqual(checkpoint["notes"], ["COMPLETED PART 1"])
            api.return_value.chat.completions.create.reset_mock()
            api.return_value.chat.completions.create.side_effect = [
                response(f"REMAINING PART {i + 1}") for i in range(1, len(parts))
            ] + [response(FINAL)]
            result = analyze_with_groq(transcript, checkpoint=checkpoint)
        calls = api.return_value.chat.completions.create.call_args_list
        self.assertEqual(len(calls), len(parts))
        self.assertIn(parts[1], calls[0].kwargs["messages"][1]["content"])
        self.assertNotIn(parts[0], calls[0].kwargs["messages"][1]["content"])
        self.assertEqual(len(checkpoint["notes"]), len(parts))
        self.assertIn("Whole video", result["title"])

    @patch("core.groq_analysis.Groq")
    def test_notes_reduction_covers_all_groups(self, api):
        notes = [f"PART {i}:" + str(i) * 6000 for i in range(4)]
        api.return_value.chat.completions.create.side_effect = [
            response("CONDENSED FIRST HALF"), response("CONDENSED SECOND HALF")
        ]
        output = _synthesize(api.return_value, notes)
        self.assertIn("CONDENSED FIRST HALF", output)
        self.assertIn("CONDENSED SECOND HALF", output)
        requests = api.return_value.chat.completions.create.call_args_list
        self.assertEqual(len(requests), 2)
        inputs = " ".join(call.kwargs["messages"][1]["content"] for call in requests)
        for i in range(4):
            self.assertIn(f"PART {i}:", inputs)


if __name__ == "__main__":
    unittest.main()
