# 🎬 AI Video Assistant

Analyze accessible YouTube captions, uploaded audio/video, or pasted transcripts. Generate a title, summary, key insights, and transcript-grounded Q&A.

## Inputs and providers

| Input | Behavior | Required credential |
| --- | --- | --- |
| YouTube URL | Read accessible captions first, then try yt-dlp audio and Groq Whisper if needed. YouTube may block *both* from cloud IPs. | Groq or Mistral for AI analysis; Groq also needed for audio fallback. |
| Upload MP3/WAV/M4A/MP4/MKV/WEBM | FFmpeg normalizes audio; Groq Whisper transcribes it. | Groq, plus Mistral only if selected for analysis. |
| Paste transcript | Process text without contacting YouTube or using Whisper. | Groq **or** Mistral. |

Choose **Groq (default when configured)** or **Mistral** under AI analysis provider. You can use Groq with only `GROQ_API_KEY`; `MISTRAL_API_KEY` is optional. When Mistral responds with HTTP 429 and Groq is configured, analysis falls back to Groq. A 429 from Groq must still be resolved against its own quota.

### Why Groq analysis uses fewer requests

For a normal-length transcript, `core/groq_analysis.py` creates the title, overview, takeaways, action items, decisions, and open questions in **one structured chat request** using `openai/gpt-oss-20b`. Q&A uses keyword-based retrieval over the **full** session transcript and a Groq chat request, with no embedding API calls. Keyword matching is simpler than semantic vector retrieval; it can miss paraphrases. Mistral remains available as an alternative with semantic embeddings and more API requests.

For transcripts exceeding 18,000 characters, the Groq *overview* is explicitly marked as based on sampled excerpts from the beginning, middle and end. It is **not** a complete-video summary. The full original transcript is preserved in the session and used for Q&A. Summaries and Q&A may be inaccurate; verify important details.

## Data flow

```text
YouTube URL ----> accessible captions? ---- yes ---> transcript + timestamps --+
                     | no                                                |
                     v                                                   |
              yt-dlp → FFmpeg → Groq Whisper -----------------------------+
                                                                         |
Uploaded media --------> FFmpeg → Groq Whisper ---------------------------+
                                                                         |
Pasted transcript --------------------------------------------------------+
                                                                         |
                                                   saved Streamlit session
                                                                         |
                                      choose Groq or Mistral for analysis
                                                                         |
                                      title + summary + structured insights
                                                                         |
                                          transcript-grounded Q&A
```

## Deploy on Streamlit Community Cloud

1. Deploy repository `alisalmann7386-crypto/AI-VIDEO-ASSISTANT`, branch `main`, entrypoint `app.py`.
2. Select **Python 3.12** in Advanced settings. To change the Python version of an existing deployed app, follow Streamlit's delete-and-redeploy procedure after recording its settings.
3. Enter your own keys in the app's private **Secrets** field; never commit real keys or cookies to GitHub:

```toml
GROQ_API_KEY = "your-real-groq-key"
# Optional if you want Mistral analysis as an alternative:
MISTRAL_API_KEY = "your-real-mistral-key"
```

4. Deploy/reboot. Test a short pasted transcript with **Groq** selected first, then an uploaded WAV, then a public YouTube URL. `requirements.txt` installs Python dependencies and `packages.txt` installs FFmpeg.

### HTTP 429: rate limit exceeded

Mistral's API enforces request and token limits at the account/organization level. The old workflow called Mistral separately for a title, multiple summaries, insights and embeddings, so even a pasted transcript could fail. The new default Groq workflow reduces that to one analysis request without embeddings; it **does not** grant extra Mistral or Groq quota. Check your provider's developer dashboard for limits, usage, available models, and billing. Avoid rapid retries against an exhausted quota.

**Your transcript is saved in `st.session_state` before analysis begins.** If any AI provider rejects the request, the transcript tab remains available and a **Retry analysis from saved transcript** button appears. Choose another provider or retry when limits permit—there is no need to download or transcribe the media again. This cache is session-local and is lost if the Streamlit session is reset or the server restarts; download the transcript as a backup.

### YouTube restrictions

Captions are attempted before audio and can use any available original-language track. YouTube may still block both captions and media downloads from the Streamlit server. In that case, use the **Paste transcript** option with YouTube's Show transcript feature, or upload media you may lawfully process. A regular YouTube Data API key or downloader flags cannot guarantee access to arbitrary video captions or audio.

### Local development

Use Python 3.12 with `ffmpeg` and `ffprobe` installed and on your PATH.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Store local keys in an ignored `.env` file. The optional command-line script is `python main.py` (audio path).

## Tests, privacy and limitations

GitHub Actions installs dependencies, compiles Python and runs offline unit tests and a Streamlit startup test. AI replies and YouTube access are **mocked** and these tests do not prove live success with your account's provider quotas, a specific video, or Streamlit Cloud.

Media is limited to two hours and pasted/caption transcripts to 120,000 characters. Streamlit may impose smaller upload limits. Temporary media and chunks are deleted after extraction; the transcript and search data persist in the user's current Streamlit session. Transcripts and relevant excerpts are sent to the chosen cloud AI provider. Only process content you are permitted to use and verify important outputs against the recording.
