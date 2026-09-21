"""Verify the app can render its initial screen without provider credentials."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class StreamlitStartupTests(unittest.TestCase):
    def test_initial_page_renders_without_api_keys(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "MISTRAL_API_KEY": ""}):
            app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertFalse(app.exception, f"Streamlit raised: {app.exception}")
        self.assertTrue(app.title)


if __name__ == "__main__":
    unittest.main()
