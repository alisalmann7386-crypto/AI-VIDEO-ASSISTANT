"""Optional command-line interface; Streamlit Cloud runs app.py instead."""
from __future__ import annotations

from pathlib import Path
import tempfile
from dotenv import load_dotenv

from utils.audio_processor import is_youtube_url, process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_insights
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    """Analyze one source and retain only transcript-derived data after cleanup."""
    if not is_youtube_url(source) and not Path(source).is_file():
        raise ValueError("Enter a valid YouTube URL or an existing local file path.")
    with tempfile.TemporaryDirectory(prefix="ai_video_cli_") as workspace:
        chunks, metadata = process_input(source, work_dir=workspace)
        result = transcribe_all(chunks, language=language)
        transcript = result["full_text"]
        title = generate_title(transcript)
        summary = summarize(transcript)
        insights = extract_insights(summary)
        rag_chain = build_rag_chain(result["segments"] or transcript)
    return {"title": title, "metadata": metadata, "transcript": transcript,
            "segments": result["segments"], "summary": summary,
            "action_items": insights["action_items"],
            "key_decisions": insights["key_decisions"],
            "open_questions": insights["open_questions"], "rag_chain": rag_chain}


if __name__ == "__main__":
    source = input("YouTube URL or local audio/video path: ").strip()
    language = input("Spoken language (english/hinglish) [english]: ").strip() or "english"
    try:
        data = run_pipeline(source, language)
        print(f"\nTITLE: {data['title']}\n\nSUMMARY:\n{data['summary']}")
        print("\nACTION ITEMS:", data["action_items"])
        print("KEY DECISIONS:", data["key_decisions"])
        print("OPEN QUESTIONS:", data["open_questions"])
        while True:
            question = input("\nAsk about this video (exit to quit): ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                break
            if question:
                print(ask_question(data["rag_chain"], question))
    except Exception as error:
        # Never print a full provider traceback containing secrets to shared logs.
        print(f"Pipeline failed ({type(error).__name__}). Check the media, API keys and provider quota.")
