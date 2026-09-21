# 🎬 AI Video Assistant

Analyze YouTube videos **using accessible captions first**, or upload audio/video or paste a transcript. Get a summary, structured insights, and transcript-grounded Q&A.

## Three input methods

| Input | How it works | What you need |
| --- | --- | --- |
| **YouTube URL** | Tries available English/Hindi captions; if unavailable, downloads audio and transcribes it. | `MISTRAL_API_KEY`; `GROQ_API_KEY` if the audio fallback is needed. |
| **Upload a file** | Converts MP3/WAV/M4A/MP4/MKV/WEBM with FFmpeg; transcribes with Groq Whisper. | Both keys. |
| **Paste transcript** | Uses your pasted text for summaries, insights and RAG without requesting YouTube or Groq. Optional YouTube link associates the source. | `MISTRAL_API_KEY` only. |

Captions preserve timestamps, when provided by YouTube; plain pasted text has no reliable timestamps. English/Hinglish transcription is not automatic translation.

## How analysis works

```text
YouTube URL ──────> public captions available? ─── yes ─> timestamped text ─┐
                      │ no                                              │
                      v                                                 │
               yt-dlp audio → FFmpeg → Groq Whisper ─────────────────────┤
                                                                        │
Uploaded media → FFmpeg → Groq Whisper ──────────────────────────────────┤
                                                                        │
Pasted transcript ───────────────────────────────────────────────────────┘
                  │
                  v
       Mistral title, summary and insights
                  │
                  v
       Mistral embeddings → in-memory semantic search
                  │
                  v
       Transcript-grounded Mistral Q&A
```

No local Whisper weights, TensorFlow, FAISS, Chroma, or persistent vector-server process is required. API usage can incur charges or rate limits.

## Deploy on Streamlit Community Cloud

1. Select repository `alisalmann7386-crypto/AI-VIDEO-ASSISTANT`, branch `main`, entrypoint `app.py`.
2. In **Advanced settings**, select **Python 3.12**. For an existing app running another Python version, Streamlit requires deleting and redeploying it to change Python; record the app subdomain and secrets first.
3. Add your own credentials to the app's **Secrets** settings; do not commit keys or browser cookies to GitHub:

```toml
GROQ_API_KEY = "your-groq-api-key"
MISTRAL_API_KEY = "your-mistral-api-key"
```

4. Deploy and test **Paste transcript** with a short sample first, then an uploaded MP3/WAV, then a public YouTube link. `requirements.txt` installs Python packages; `packages.txt` installs system FFmpeg.

## If YouTube says "blocked this download"

A blocked YouTube request is **not the same** as a failed Streamlit installation. The app now tries captions before downloading media, but caption requests can also be blocked from cloud-provider IPs, or captions might be disabled for the video. It is not possible to guarantee that any YouTube URL will work from Streamlit Community Cloud.

**Fallback without downloading the video:** Open the video on YouTube and, if **Show transcript** is available, copy its text. In the app choose **Paste transcript**, paste the text, optionally add the original YouTube link, and click **Analyze content**. This route doesn't need access to YouTube from the Streamlit server. Plain pasted text won't have timestamped Q&A citations.

For media you own or have permission to process, **Upload a file** is another option. Do not publish account cookies or API credentials as a workaround.

### Local development

Use Python 3.12 and install `ffmpeg` and `ffprobe` on your PATH.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Store keys locally in a `.env` file that is ignored by Git. The optional CLI is `python main.py` (media transcription path).

## Tests and limitations

GitHub Actions installs packages, compiles Python, runs offline unit tests and checks Streamlit startup without provider credentials. Captions in tests are mocked. These tests **do not establish** that a particular video, account, YouTube region, Groq quota, or Mistral quota is accessible from your deployed server.

The app limits media to two hours and pasted/caption transcripts to 120,000 characters. Large videos may exceed Streamlit upload limits. Uploaded media and extracted audio are deleted after processing; the transcript and RAG index stay in the current Streamlit session. Uploaded media, captions and pasted transcripts are sent to the configured AI providers for processing. Only process content you have permission to use, and verify important summary claims against the original media.
