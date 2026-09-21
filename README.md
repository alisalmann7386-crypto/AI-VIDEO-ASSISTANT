# 🎬 AI Video Assistant

Turn a public YouTube video or an uploaded audio/video file into a transcript, summary, structured insights and transcript-grounded Q&A.

## Features

- **Inputs:** YouTube watch/Shorts links, or MP3, WAV, M4A, MP4, MKV and WEBM uploads.
- **Transcription:** Groq-hosted multilingual Whisper (English or Hinglish), with segment timestamps when provided by the API.
- **Summaries:** Mistral AI generates an overview, takeaways and supported action items / open questions.
- **Q&A:** Mistral embeddings + an in-memory cosine-similarity index retrieve transcript excerpts, then Mistral answers from them.
- **Exports:** Download the summary and transcript as text.
- **Privacy:** Uploaded media and intermediate MP3 chunks live in a temporary directory and are deleted after processing; transcript and index remain in your Streamlit session until reset. Media and transcript are sent to the configured cloud AI providers.

## Architecture

```text
YouTube URL or uploaded media
          |
          v
    yt-dlp (YouTube only)
          |
          v
 ffmpeg audio normalization / 10-minute chunks
          |
          v
   Groq Whisper transcription
          |
          +----> timestamped transcript -------------------+
          |                                                |
          v                                                v
 Mistral title + hierarchical summary            Mistral embeddings
          |                                                |
          v                                                v
   Structured insights                         In-memory cosine retrieval
                                                           |
                                         User question + transcript excerpts
                                                           |
                                                           v
                                                 Mistral grounded answer
```

This application does **not** need TensorFlow, a local Whisper model, FAISS, ChromaDB, or a persistent vector server. It does need internet access and working API keys for Groq and Mistral AI. API usage may incur provider charges and rate limits.

## Deploy on Streamlit Community Cloud

1. Select repository `alisalmann7386-crypto/AI-VIDEO-ASSISTANT`, branch `main`, and entrypoint **`app.py`**.
2. In **Advanced settings**, select **Python 3.12**. `runtime.txt` is only a hint; Community Cloud selects Python using Advanced settings. To change Python for an already deployed app, follow Streamlit's official delete-and-redeploy procedure after recording your app URL and secrets.
3. In the app **Secrets** field, add the following TOML with your own real API keys (do **not** commit it to GitHub):

```toml
GROQ_API_KEY = "your-groq-api-key"
MISTRAL_API_KEY = "your-mistral-api-key"
```

4. Deploy. `requirements.txt` installs Python packages; `packages.txt` installs FFmpeg. The app should load even without credentials, but analysis requires both keys.
5. Test using a **short uploaded WAV or MP3 file first**. Then try a public YouTube video.

> ⚠️ **YouTube caveat:** YouTube may refuse downloads from cloud-hosting IPs (HTTP 403 or bot checks). No yt-dlp client flags can guarantee access. This is an upstream restriction, not a Streamlit deployment error. Use the file-upload path if a video is blocked. Do not commit YouTube session cookies or API keys.

> **Upload limits:** Streamlit's default per-file upload limit may be lower than a large video. Short audio files are the most reliable choice on free hosting. Maximum duration enforced by the app is two hours. English and Hinglish are *transcribed*, not automatically translated to English.

### Local development

Use Python 3.12 and make sure `ffmpeg` and `ffprobe` are installed and on your `PATH`.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a local `.env` (ignored by Git) containing `GROQ_API_KEY` and `MISTRAL_API_KEY`, then run:

```bash
streamlit run app.py
```

The optional command-line interface is `python main.py`.

## Automated checks

GitHub Actions runs a clean Python 3.12 installation, compiles Python files, and runs offline regression tests (`python -m unittest discover -s tests -v`). These checks do not exercise paid LLM APIs, authenticate to YouTube, or verify a live Streamlit deployment.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `GROQ_API_KEY` / `MISTRAL_API_KEY` missing | Set both as root-level TOML keys in Streamlit app Secrets; restart the app. |
| `ffmpeg/ffprobe is missing` | Ensure root `packages.txt` contains `ffmpeg` and rebuild the application. |
| YouTube HTTP 403 | Use file upload; cloud-IP blocks cannot be guaranteed away. |
| Provider 401 / 429 | Check API key, model access, provider quota and billing directly with the provider. |
| App uses the wrong Python version | Choose Python 3.12 in Advanced settings; use Streamlit's redeploy procedure if changing an existing app. |
| Very long recording fails or is expensive | Test a short recording first; transcript summarization and embeddings require external API requests. |

## Responsible use

Automated summaries can miss details; confirm important information against the original recording. Only upload media you are permitted to process. Do not include sensitive recordings unless your organization's data-handling rules permit sending them to the configured cloud providers.
