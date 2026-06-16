Quick setup and run path, in order.

First, the heavy dependencies. pyannote needs a HuggingFace token because the diarization model is gated:Here's the actual sequence, step by step.

**1. Install dependencies**

```bash
pip install openai-whisper pyannote.audio torch torchaudio \
            spacy keybert nltk numpy pandas networkx \
            plotly matplotlib fastapi uvicorn sounddevice \
            sqlalchemy psycopg2-binary python-multipart \
            reportlab jinja2 soundfile

python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

If you're on a machine without a GPU, this still works — it'll just run on CPU and be slower (Whisper `base` on CPU for a 15-min audio file is roughly real-time-ish, a few minutes).

**2. Set up the HuggingFace token for diarization**

pyannote's speaker-diarization model is gated. Two things needed:

1. Create an account at huggingface.co, generate an access token (Settings → Access Tokens → create one with "read" permission).
2. Visit `huggingface.co/pyannote/speaker-diarization-3.1` and `huggingface.co/pyannote/segmentation-3.0`, accept the user agreement on both (they're free, just need a click-through).

Then either:

```bash
huggingface-cli login
# paste token when prompted
```

or set it as an env var, which is what the code reads by default:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
```

Without this step, Phase 1 diarization will throw an auth error.

**3. Run on a recorded audio file**

```bash
python edgegd.py analyze --audio session.wav --topic "AI in Education" --speakers 4
```

`--speakers 4` tells pyannote how many distinct speakers to expect — pass this if you know the count (your placement-cell use case usually does), it noticeably improves diarization accuracy. Omit it if unknown and pyannote will estimate.

Supported audio formats: anything ffmpeg can read (wav, mp3, m4a). If you don't have ffmpeg installed, Whisper will fail on non-wav files — `apt install ffmpeg` or `brew install ffmpeg` depending on OS.

**4. Check the output**

After it finishes, look in `edgegd_output/`:

```
edgegd_output/
├── result_<session_id>.json
├── report_<session_id>.pdf
├── participation_pie.html
├── speaking_time_bar.html
├── timeline.html
├── network_graph.html
└── summary.png
```

Open the `.html` files directly in a browser — no server needed. Open the PDF normally.

**5. Real-time mode (Phase 10)**

```bash
python edgegd.py live --duration 900 --speakers 4
```

This needs a working microphone and `sounddevice` to detect your audio device correctly. `--duration 900` runs for 15 minutes; omit it to run until Ctrl+C. Output is a live-updating terminal dashboard, not a file.

**6. API server (optional)**

```bash
uvicorn edgegd:app --reload
```

Then POST an audio file to `http://localhost:8000/analyze` (e.g. via curl, Postman, or a frontend) — returns the same JSON as the CLI plus downloadable PDF via `/report/{session_id}`.

---

**A note on first run**: the first time you run `analyze`, expect a pause while Whisper downloads its model weights (~150MB for `base`) and pyannote downloads its pipeline (~a few hundred MB). Both cache locally after that, so subsequent runs are fast to start.

For your first test, I'd suggest grabbing a short (2-3 min) recording with 2-3 speakers — easier to sanity-check the metrics against what you actually heard, before running it on a full mock GD session.