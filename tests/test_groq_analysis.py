"""Groq analysis regression tests; all network calls are mocked."""
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.groq_analysis import (
    GROQ_CHAT_MODEL, analysis_excerpt, analyze_with_groq, build_groq_rag,
    is_rate_limited,
)


class GroqAnalysisTests(unittest.TestCase):
    def test_short_transcript_is_not_truncated(self):
        content, notice = analysis_excerpt("A short meeting about project risks.")
        self.assertEqual(content, "A short meeting about project risks.")
        self.assertFalse(notice)

    def test_long_transcript_has_explicit_sampling_disclosure(self):
        text = "START" + "a" * 25000 + "MIDDLE" + "b" * 25000 + "END"
        excerpt, notice = analysis_excerpt(text)
        self.assertIn("START", excerpt)
        self.assertIn("END", excerpt)
        self.assertIn("sampled", notice)
        self.assertLessEqual(len(excerpt), 18000)

    @patch("core.groq_analysis.Groq")
    def test_one_structured_call_generates_title_summary_insights(self, client_class):
        response = {"title": "Release planning", "overview": "The team planned a release.",
                    "key_takeaways": ["Testing is required"],
                    "action_items": ["Run tests"], "key_decisions": [], "open_questions": []}
        client_class.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            data = analyze_with_groq("Release planning discussion with testing assigned.")
        self.assertEqual(data["title"], "Release planning")
        self.assertIn("Testing is required", data["summary"])
        self.assertEqual(data["insights"]["action_items"], ["Run tests"])
        self.assertEqual(data["insights"]["key_decisions"], ["None discussed"])
        self.assertFalse(data["coverage_note"])
        call = client_class.return_value.chat.completions.create
        call.assert_called_once()
        self.assertEqual(call.call_args.kwargs["model"], GROQ_CHAT_MODEL)
        self.assertTrue(call.call_args.kwargs["response_format"]["json_schema"]["strict"])

    def test_rate_limit_identification(self):
        self.assertTrue(is_rate_limited(SimpleNamespace(status_code=429)))
        self.assertTrue(is_rate_limited(SimpleNamespace(response=SimpleNamespace(status_code=429))))
        self.assertFalse(is_rate_limited(SimpleNamespace(status_code=401)))

    @patch("core.groq_analysis.Groq")
    def test_local_retrieval_and_one_question_request(self, client_class):
        client_class.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="The deadline is Friday [00:10–00:14]."))])
        rag = build_groq_rag([
            {"start": "00:00", "end": "00:05", "text": "The team greeted each other."},
            {"start": "00:10", "end": "00:14", "text": "The project deadline is Friday."},
        ])
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            answer = rag.invoke("What is the project deadline?")
        self.assertIn("Friday", answer)
        client_class.return_value.chat.completions.create.assert_called_once()
        request = client_class.return_value.chat.completions.create.call_args.kwargs
        self.assertIn("deadline", request["messages"][1]["content"])

    @patch("core.groq_analysis.Groq")
    def test_keyword_retrieval_does_not_call_api_without_match(self, client_class):
        rag = build_groq_rag("The presenter talks about gardening and sunlight.")
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            self.assertIn("couldn't find", rag.invoke("quantum spacecraft"))
        client_class.assert_not_called()

    def test_groq_key_is_required(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                analyze_with_groq("A valid transcript")


if __name__ == "__main__":
    unittest.main()
