"""
EdgeGD web backend.

Serves the upload UI and runs the Phase 1-9 analysis pipeline from python.py
as a background job, so the browser can poll progress instead of holding a
request open for the length of a recording.

------------------------------------------------------------------
RUN
------------------------------------------------------------------
    cd edgeGD
    uvicorn server:app --reload --port 8000

Then open http://localhost:8000

Analysis is CPU-bound and serialised through a single worker, so two uploads
queue rather than fight over the models. Whisper and pyannote are loaded once
and kept warm for the life of the process.
"""

import os
import re
import sys
import json
import time
import shutil
import uuid
import threading
import subprocess
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

# --- make `python.py` importable regardless of the caller's cwd -------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# pyannote's diarization repo is gated. If there is no token but the weights
# are already in the local HF cache, force offline mode so from_pretrained
# reads the cache instead of failing a 401 against the hub.
if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
    _cached = os.path.expanduser(
        "~/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1")
    if os.path.isdir(_cached) and not os.path.exists(
            os.path.expanduser("~/.cache/huggingface/token")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

import python as edgegd  # noqa: E402  (the pipeline module, edgeGD/python.py)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402


# ===================================================================
# LAYOUT ON DISK
# ===================================================================

OUTPUT_ROOT = os.path.abspath(edgegd.Config.OUTPUT_DIR)
SESSIONS_DIR = os.path.join(OUTPUT_ROOT, "sessions")
UPLOAD_DIR = os.path.join(OUTPUT_ROOT, "uploads")
WEB_DIR = os.path.join(HERE, "web")

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}$")
CHART_FILES = {
    "participation_pie": "participation_pie.html",
    "speaking_time_bar": "speaking_time_bar.html",
    "timeline": "timeline.html",
    "network_graph": "network_graph.html",
    "summary": "summary.png",
}


def session_dir(session_id: str) -> str:
    """Resolve a session's directory, rejecting anything that isn't an id."""
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Malformed session id")
    return os.path.join(SESSIONS_DIR, session_id)


# ===================================================================
# AUDIO PREP
# ===================================================================

def probe_duration(path: str) -> Optional[float]:
    """Media length in seconds via ffprobe, or None if it can't be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            check=True, capture_output=True, text=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def to_analysis_wav(src: str, dest: str, limit_sec: Optional[float] = None) -> None:
    """
    Decode any upload (video or audio) to the 16 kHz mono WAV both Whisper and
    pyannote want. Doing it once here avoids each stage re-decoding the file,
    and `limit_sec` lets the UI analyse only the opening minutes of a long
    recording.
    """
    cmd = ["ffmpeg", "-y", "-i", src]
    if limit_sec:
        cmd += ["-t", str(limit_sec)]
    cmd += ["-ar", str(edgegd.Config.SAMPLE_RATE), "-ac", "1", "-vn", dest]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed, so the upload can't be decoded. "
            "Install it with `brew install ffmpeg` (macOS) or "
            "`apt install ffmpeg` (Linux).")
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(
            "ffmpeg could not decode this file. " + (tail[-1] if tail else ""))


# ===================================================================
# JOBS
# ===================================================================

@dataclass
class Job:
    job_id: str
    filename: str
    topic: str
    num_speakers: Optional[int] = None
    limit_sec: Optional[float] = None
    source_duration: Optional[float] = None
    analysed_duration: Optional[float] = None

    status: str = "queued"            # queued | running | done | error
    phase: str = "Waiting for the analyser"
    progress: int = 0
    session_id: Optional[str] = None
    error: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


_jobs: Dict[str, Job] = {}
_jobs_lock = threading.Lock()

# One worker: the models are large and the work is CPU-bound, so a second
# concurrent analysis would only make both slower.
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edgegd")

# Warm models, created on first use and reused for every later session.
_engines: Dict[str, Any] = {}
_engines_lock = threading.Lock()


def get_engines():
    with _engines_lock:
        if not _engines:
            _engines["stt"] = edgegd.SpeechToTextEngine()
            _engines["diarizer"] = edgegd.SpeakerDiarizer()
        return _engines["stt"], _engines["diarizer"]


def update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            for k, v in fields.items():
                setattr(job, k, v)


def run_job(job_id: str, upload_path: str) -> None:
    """Worker body: prepare audio, run Phases 1-9, record where it landed."""
    with _jobs_lock:
        job = _jobs[job_id]
        topic = job.topic
        num_speakers = job.num_speakers
        limit_sec = job.limit_sec
        original_name = job.filename

    update_job(job_id, status="running", started_at=time.time(),
               phase="Decoding audio", progress=1)

    session_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "audio.wav")

    try:
        to_analysis_wav(upload_path, wav_path, limit_sec)
        analysed = probe_duration(wav_path)
        update_job(job_id, analysed_duration=analysed,
                   phase="Loading models", progress=3)

        stt, diarizer = get_engines()

        def on_progress(label: str, pct: int):
            update_job(job_id, phase=label, progress=max(3, min(99, pct)))

        result = edgegd.analyze_session(
            audio_path=wav_path,
            topic=topic,
            num_speakers=num_speakers,
            output_dir=out_dir,
            progress=on_progress,
            stt=stt,
            diarizer=diarizer,
            session_id=session_id,
            total_duration_sec=analysed,
        )

        # Record what produced this session, alongside its result JSON.
        with open(os.path.join(out_dir, "source.json"), "w") as f:
            json.dump({
                "session_id": session_id,
                "original_filename": original_name,
                "source_duration_sec": probe_duration(upload_path),
                "analysed_duration_sec": analysed,
                "limit_sec": limit_sec,
                "requested_speakers": num_speakers,
                "topic": topic,
                "analysed_at": result.date,
            }, f, indent=2)

        update_job(job_id, status="done", progress=100,
                   phase="Report ready", session_id=session_id,
                   finished_at=time.time())

    except Exception as e:  # surfaced verbatim in the UI - it's a dev tool
        update_job(job_id, status="error", error=f"{type(e).__name__}: {e}",
                   phase="Failed", finished_at=time.time())
        shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)     # the source upload is kept; this is derived
        if os.path.exists(upload_path):
            os.remove(upload_path)


# ===================================================================
# APP
# ===================================================================

app = FastAPI(title="EdgeGD Web", version="1.0")


@app.post("/api/analyze")
async def api_analyze(
    audio: UploadFile = File(...),
    topic: str = Form("Group Discussion"),
    num_speakers: Optional[str] = Form(None),
    limit_minutes: Optional[str] = Form(None),
):
    """Accept an upload and queue it. Returns immediately with a job id."""
    job_id = uuid.uuid4().hex[:12]
    safe_name = os.path.basename(audio.filename or "upload")
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}__{safe_name}")

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    def as_number(raw, cast, label):
        if raw in (None, "", "auto"):
            return None
        try:
            value = cast(raw)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{label} must be a number")
        return value if value > 0 else None

    speakers = as_number(num_speakers, int, "Speaker count")
    limit_min = as_number(limit_minutes, float, "Minute limit")

    duration = probe_duration(upload_path)
    if duration is None:
        os.remove(upload_path)
        raise HTTPException(
            status_code=400,
            detail="That file has no readable audio track.")

    job = Job(
        job_id=job_id,
        filename=safe_name,
        topic=topic.strip() or "Group Discussion",
        num_speakers=speakers,
        limit_sec=limit_min * 60 if limit_min else None,
        source_duration=duration,
    )
    with _jobs_lock:
        _jobs[job_id] = job

    _pool.submit(run_job, job_id, upload_path)
    return asdict(job)


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return asdict(job)


@app.get("/api/jobs")
async def api_jobs():
    with _jobs_lock:
        jobs = [asdict(j) for j in _jobs.values()]
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return {"jobs": jobs}


def read_session(session_id: str) -> Dict[str, Any]:
    path = os.path.join(session_dir(session_id), f"result_{session_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No such session")
    with open(path) as f:
        return json.load(f)


@app.get("/api/sessions")
async def api_sessions():
    """Index of completed sessions, newest first."""
    items: List[Dict[str, Any]] = []
    for sid in os.listdir(SESSIONS_DIR):
        result_path = os.path.join(SESSIONS_DIR, sid, f"result_{sid}.json")
        if not os.path.exists(result_path):
            continue
        try:
            with open(result_path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        source = {}
        source_path = os.path.join(SESSIONS_DIR, sid, "source.json")
        if os.path.exists(source_path):
            try:
                with open(source_path) as f:
                    source = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        items.append({
            "session_id": d["session_id"],
            "date": d["date"],
            "topic": d["topic"],
            "duration_sec": d["total_duration_sec"],
            "participants": len(d["participants"]),
            "cohesion_score": d["group_cohesion_score"],
            "cohesion_label": d["group_cohesion_label"],
            "filename": source.get("original_filename"),
            "mtime": os.path.getmtime(result_path),
        })

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"sessions": items}


@app.get("/api/sessions/{session_id}")
async def api_session(session_id: str):
    data = read_session(session_id)
    source_path = os.path.join(session_dir(session_id), "source.json")
    if os.path.exists(source_path):
        try:
            with open(source_path) as f:
                data["source"] = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    data["artifacts"] = [
        name for name, fname in CHART_FILES.items()
        if os.path.exists(os.path.join(session_dir(session_id), fname))
    ]
    # The UI explains thresholds in its notes; read them from Config so the
    # two can't drift apart.
    data["config"] = {
        "silent_pct": edgegd.Config.SILENT_THRESHOLD_PCT,
        "moderate_pct": edgegd.Config.MODERATE_THRESHOLD_PCT,
        "dominant_pct": edgegd.Config.DOMINANT_THRESHOLD_PCT,
        "pace_slow_wpm": edgegd.Config.PACE_SLOW_WPM,
        "pace_fast_wpm": edgegd.Config.PACE_FAST_WPM,
        "long_pause_sec": edgegd.Config.LONG_PAUSE_SEC,
        "whisper_model": edgegd.Config.WHISPER_MODEL,
    }
    return data


@app.get("/api/sessions/{session_id}/report.pdf")
async def api_session_pdf(session_id: str):
    path = os.path.join(session_dir(session_id), f"report_{session_id}.pdf")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No PDF for this session")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"edgegd_report_{session_id}.pdf")


@app.get("/api/sessions/{session_id}/chart/{name}")
async def api_session_chart(session_id: str, name: str):
    """The Plotly/Matplotlib artifacts Phase 8 writes."""
    fname = CHART_FILES.get(name)
    if not fname:
        raise HTTPException(status_code=404, detail="Unknown chart")
    path = os.path.join(session_dir(session_id), fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chart not generated")
    media = "image/png" if fname.endswith(".png") else "text/html"
    return FileResponse(path, media_type=media)


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    path = session_dir(session_id)
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="No such session")
    shutil.rmtree(path)
    return {"deleted": session_id}


@app.get("/api/health")
async def api_health():
    """What the pipeline can actually do in this environment."""
    def importable(module: str) -> bool:
        import importlib.util
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "whisper_model": edgegd.Config.WHISPER_MODEL,
        "diarization_model": edgegd.Config.DIARIZATION_MODEL,
        "hf_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "keybert": importable("keybert"),
        "models_warm": bool(_engines),
    }


# --- static frontend -------------------------------------------------------

if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="web/index.html is missing")
    with open(path) as f:
        return HTMLResponse(f.read())
