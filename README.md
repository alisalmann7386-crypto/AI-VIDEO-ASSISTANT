# 🎬 AI Video Assistant

Analyze accessible YouTube captions, uploaded audio/video, or pasted transcripts. Get a summary of the **entire supplied transcript**, key insights, and transcript-grounded Q&A.

## How input and transcription work

| Input | Processing | Required key |
| --- | --- | --- |
| YouTube URL with accessible captions | Fetch YouTube's caption text and timestamps. **Do not run Whisper again**: the captions are already a transcript. | Groq or Mistral for analysis. |
| YouTube URL without accessible captions | Attempt an audio download, normalize with FFmpeg, split audio and transcribe chunks using Groq Whisper. Cloud IPs may be blocked. | Groq for transcription; Groq or Mistral for analysis. |
| Uploaded audio/video | FFmpeg audio extraction and audio chunking, then Groq Whisper speech-to-text. | Groq for transcription; Groq or Mistral for analysis. |
| Pasted transcript | Use the supplied text directly. No YouTube access, audio download, or Whisper transcription. | Groq or Mistral for analysis. |

**Audio chunking and transcript chunking are different:** audio is split only if the app must transcribe speech. All long text transcripts (including downloaded YouTube captions) are split for summarization. YouTube may deny access to captions *and* audio from Streamlit Community Cloud; a code change cannot guarantee access to arbitrary videos.

## Full-transcript summarization

Choose **Groq** (default if `GROQ_API_KEY` is configured) or **Mistral** in the sidebar.

For Groq, a transcript of 18,000 characters or less is analyzed in one structured call to `openai/gpt-oss-20b`. Larger transcripts use a **lossless, ordered map-and-reduce pipeline**:

```text
Full caption / Whisper / pasted transcript
                   |
                   v
       Split ALL text into ~8,500-character chunks
                   |
                   v
       Summarize EVERY chunk in sequence
                   |
                   v
       Condense all chunk notes if needed
                   |
                   v
       Final title + overview + structured insights
                   |
                   v
       Q&A retrieves from the original FULL transcript
```

No start/middle/end sampling is used in the production summary anymore. A summary can still omit fine details or contain mistakes; verify critical points against the full transcript.

**Quota tradeoff:** processing the full transcript requires more Groq calls than the previous sampled overview. Each successfully summarized chunk is saved in the current Streamlit session, so after an HTTP 429 the **Retry analysis from saved transcript** button resumes from the next unsummarized chunk rather than repeating completed calls. The transcript is also preserved. This does not bypass account rate or token limits. Session state is lost if the tab/session resets or the server restarts, so download important transcripts as a backup.

Groq Q&A uses keyword retrieval over the original complete transcript with no embedding requests. It can miss paraphrases and synonyms; Mistral's optional semantic embeddings provide an alternative but use additional API calls. When Mistral returns HTTP 429 and Groq is configured, the app attempts Groq analysis instead.

## Deployment on Streamlit Community Cloud

1. Choose repository `alisalmann7386-crypto/AI-VIDEO-ASSISTANT`, branch `main`, entrypoint `app.py`.
2. Select Python **3.12** in Advanced settings. Changing Python for an existing app may require deleting and redeploying it; record the app URL and secrets first.
3. Add **your own keys** in private Streamlit Secrets. Never upload real credentials or browser cookies to GitHub:

```toml
GROQ_API_KEY = "your-groq-api-key"
# Optional, if you also want Mistral:
MISTRAL_API_KEY = "your-mistral-api-key"
```

4. Deploy or reboot. Try a short pasted transcript with Groq first, then a public YouTube video with captions. `requirements.txt` supplies Python dependencies; `packages.txt` installs FFmpeg.

## If an API returns HTTP 429

Both Groq and Mistral have account-specific request and token quotas. A 429 is a provider response, not a Streamlit installation error. Avoid rapidly retrying an exhausted quota; consult the provider dashboard. The app preserves its transcript and any completed Groq text chunks in the same Streamlit session for a later retry. Switching from Mistral to Groq also requires a valid Groq key and available Groq quota.

## Local development

Install Python 3.12, `ffmpeg` and `ffprobe`, then run:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Store keys locally in a git-ignored `.env` file. Optional CLI: `python main.py` (audio path).

## Tests, privacy and limits

GitHub Actions installs dependencies, compiles Python, runs offline unit tests and checks Streamlit startup. Caption and AI provider calls are mocked in CI. Passing tests **does not verify a particular live YouTube video, Groq quota, Mistral quota or deployed Streamlit session**.

The app limits media to two hours and pasted/caption transcripts to 120,000 characters; Streamlit may impose lower upload limits. Temporary media and extracted audio are removed after processing. Transcript data, search passages and resumable notes exist in the active user's Streamlit session. Captions, transcripts or selected excerpts are sent to the chosen cloud AI provider. Only process media you have permission to use.
