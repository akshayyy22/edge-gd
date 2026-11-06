"""
EdgeGD: Intelligent Privacy-Preserving Real-Time Group Discussion
Analysis and Communication Assessment System

Implements Phases 1-10 from the EdgeGD developer documentation.

------------------------------------------------------------------
INSTALLATION
------------------------------------------------------------------
pip install openai-whisper pyannote.audio torch torchaudio \
            spacy keybert nltk numpy pandas networkx \
            plotly matplotlib fastapi uvicorn sounddevice \
            sqlalchemy psycopg2-binary python-multipart \
            reportlab jinja2

python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

pyannote.audio requires a HuggingFace token (gated model):
  huggingface-cli login
  -> accept terms at huggingface.co/pyannote/speaker-diarization-3.1

------------------------------------------------------------------
RUN
------------------------------------------------------------------
# CLI batch mode (Phases 1-9 on a recorded file)
python edgegd.py analyze --audio session.wav --topic "AI in Education"

# Real-time mode (Phase 10)
python edgegd.py live --duration 900

# API server
uvicorn edgegd:app --reload
------------------------------------------------------------------
"""

import os
import re
import json
import uuid
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx

# ===================================================================
# CONFIGURATION
# ===================================================================

class Config:
    WHISPER_MODEL = "base"  # tiny/base/small/medium/large
    DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

    SAMPLE_RATE = 16000

    # Module 5/6 thresholds
    SILENT_THRESHOLD_PCT = 10.0      # below this => silent
    MODERATE_THRESHOLD_PCT = 20.0    # below this => moderate, else active
    DOMINANT_THRESHOLD_PCT = 40.0    # at/above this => dominant

    # Module 7 - interruption detection
    INTERRUPTION_GAP_SEC = 0.5       # speaker B starts within this window
                                       # of speaker A's segment while A is
                                       # still "active" => interruption
    MIN_INTERRUPTING_SEGMENT_SEC = 0.3

    # Module 8 - overlap
    MIN_OVERLAP_SEC = 0.2

    # Module 10 - pace
    PACE_SLOW_WPM = 110
    PACE_FAST_WPM = 160

    # Module 11 - filler words
    FILLER_WORDS = {
        "um", "umm", "uh", "uhh", "like", "actually", "basically",
        "literally", "you know", "i mean", "sort of", "kind of",
        "so", "well", "right", "okay so"
    }

    # Module 12 - pause analysis
    LONG_PAUSE_SEC = 2.0
    HESITATION_PAUSE_SEC = 0.5

    # Module 15 - topic drift
    TOPIC_SIMILARITY_THRESHOLD = 0.25

    # Phase 10 - real-time
    LIVE_CHUNK_SEC = 5
    LIVE_BUFFER_SEC = 30

    OUTPUT_DIR = "edgegd_output"
    DB_URL = os.environ.get("EDGEGD_DB_URL", "sqlite:///edgegd.db")


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ===================================================================
# DATA MODELS (shared across phases)
# ===================================================================

@dataclass
class TranscriptSegment:
    speaker: str
    start: float
    end: float
    text: str
    words: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ParticipantMetrics:
    participant_id: str
    speaking_time_sec: float = 0.0
    participation_pct: float = 0.0
    classification: str = "Unknown"          # Silent/Moderate/Active/Dominant
    interruptions_made: int = 0
    interruptions_received: int = 0
    overlap_time_sec: float = 0.0
    word_count: int = 0
    avg_wpm: float = 0.0
    pace_label: str = "Normal"
    filler_count: int = 0
    filler_density_pct: float = 0.0
    long_pauses: int = 0
    hesitation_pauses: int = 0
    keywords: List[str] = field(default_factory=list)
    communication_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


@dataclass
class SessionResult:
    session_id: str
    date: str
    topic: str
    total_duration_sec: float
    transcript: List[TranscriptSegment]
    participants: Dict[str, ParticipantMetrics]
    interruption_matrix: Dict[str, Dict[str, int]]
    overlaps: List[Dict[str, Any]]
    turn_taking_score: float
    turn_sequence: List[str]
    topic_timeline: List[Dict[str, Any]]
    topic_drift_events: List[Dict[str, Any]]
    response_graph_edges: List[Tuple[str, str, int]]
    participation_balance_score: float
    group_cohesion_score: float
    group_cohesion_label: str


# ===================================================================
# PHASE 1 - SPEECH PROCESSING FOUNDATION
# ===================================================================

class SpeechToTextEngine:
    """Module 1: Whisper-based speech-to-text with word timestamps."""

    def __init__(self, model_size: str = Config.WHISPER_MODEL):
        self.model_size = model_size
        self._model = None

    def _load(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Returns whisper result dict with word-level timestamps."""
        model = self._load()
        result = model.transcribe(audio_path, word_timestamps=True, verbose=False)
        return result


class SpeakerDiarizer:
    """Module 2: pyannote.audio speaker diarization."""

    def __init__(self, model_name: str = Config.DIARIZATION_MODEL,
                 hf_token: Optional[str] = Config.HF_TOKEN):
        self.model_name = model_name
        self.hf_token = hf_token
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                self.model_name, token=self.hf_token
            )
        return self._pipeline

    @staticmethod
    def _prepare_audio(audio_path: str) -> Dict[str, Any]:
        """Load audio as a pyannote-compatible dict (avoids torchcodec)."""
        import subprocess
        import tempfile
        import torch
        import soundfile as sf

        path = audio_path
        tmp_path = None
        if not audio_path.lower().endswith((".wav", ".flac")):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", str(Config.SAMPLE_RATE),
                 "-ac", "1", "-vn", tmp_path],
                check=True, capture_output=True,
            )
            path = tmp_path

        try:
            data, sr = sf.read(path, dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T.contiguous()
            return {"waveform": waveform, "sample_rate": sr}
        finally:
            if tmp_path:
                os.remove(tmp_path)

    def diarize(self, audio_path: str, num_speakers: Optional[int] = None
                 ) -> List[Dict[str, Any]]:
        """Returns list of {speaker, start, end} segments."""
        pipeline = self._load()
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        audio = self._prepare_audio(audio_path)
        diarization = pipeline(audio, **kwargs)
        # pyannote.audio >= 4 wraps the Annotation in a DiarizeOutput
        annotation = getattr(diarization, "speaker_diarization", diarization)

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
            })
        return segments


class TranscriptAligner:
    """
    Combines Whisper word-level output with diarization segments
    to produce speaker-attributed transcript segments (Phase 1 deliverable).
    """

    @staticmethod
    def align(whisper_result: Dict[str, Any],
              diarization_segments: List[Dict[str, Any]]
              ) -> List[TranscriptSegment]:

        # Flatten whisper words with absolute timestamps
        all_words = []
        for seg in whisper_result.get("segments", []):
            for w in seg.get("words", []):
                all_words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"],
                })

        if not diarization_segments:
            # No diarization available -> single speaker fallback
            text = " ".join(w["word"] for w in all_words)
            start = all_words[0]["start"] if all_words else 0.0
            end = all_words[-1]["end"] if all_words else 0.0
            return [TranscriptSegment("Speaker_00", start, end, text, all_words)]

        # Sort diarization segments by start time
        diar_segments = sorted(diarization_segments, key=lambda s: s["start"])

        # Assign each word to the diarization segment with max overlap
        def best_speaker_for_word(word: Dict[str, Any]) -> str:
            wmid = (word["start"] + word["end"]) / 2.0
            best_speaker, best_overlap = "Speaker_00", -1.0
            for d in diar_segments:
                if d["start"] <= wmid <= d["end"]:
                    return d["speaker"]
                overlap = min(word["end"], d["end"]) - max(word["start"], d["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d["speaker"]
            return best_speaker

        # Build contiguous transcript segments by merging consecutive
        # words assigned to the same speaker
        segments: List[TranscriptSegment] = []
        current_speaker = None
        current_words: List[Dict[str, Any]] = []

        for w in all_words:
            sp = best_speaker_for_word(w)
            if sp != current_speaker and current_words:
                segments.append(TranscriptAligner._build_segment(
                    current_speaker, current_words))
                current_words = []
            current_speaker = sp
            current_words.append(w)

        if current_words:
            segments.append(TranscriptAligner._build_segment(
                current_speaker, current_words))

        return segments

    @staticmethod
    def _build_segment(speaker: str, words: List[Dict[str, Any]]) -> TranscriptSegment:
        text = " ".join(w["word"] for w in words)
        return TranscriptSegment(
            speaker=speaker,
            start=words[0]["start"],
            end=words[-1]["end"],
            text=text,
            words=words,
        )


def run_phase1(audio_path: str, num_speakers: Optional[int] = None,
               stt: Optional[SpeechToTextEngine] = None,
               diarizer: Optional[SpeakerDiarizer] = None,
               progress: Optional[Any] = None
               ) -> List[TranscriptSegment]:
    """
    Phase 1 pipeline: audio -> speaker-attributed transcript.

    `stt` / `diarizer` may be passed in so a long-running server can keep
    the loaded models warm across sessions instead of reloading per run.
    `progress` is an optional callable(label, pct) for UI reporting.
    """
    stt = stt or SpeechToTextEngine()
    diarizer = diarizer or SpeakerDiarizer()

    if progress:
        progress("Transcribing speech (Whisper)", 5)
    whisper_result = stt.transcribe(audio_path)

    if progress:
        progress("Identifying speakers (pyannote)", 45)
    diar_segments = diarizer.diarize(audio_path, num_speakers=num_speakers)

    if progress:
        progress("Aligning words to speakers", 70)
    transcript = TranscriptAligner.align(whisper_result, diar_segments)
    return transcript


# ===================================================================
# PHASE 2 - BASIC DISCUSSION METRICS
# ===================================================================

class SpeakingTimeAnalyzer:
    """Module 3: total speaking time per participant."""

    @staticmethod
    def compute(transcript: List[TranscriptSegment]) -> Dict[str, float]:
        totals: Dict[str, float] = defaultdict(float)
        for seg in transcript:
            totals[seg.speaker] += seg.duration
        return dict(totals)


class ParticipationAnalyzer:
    """Module 4: participation percentage."""

    @staticmethod
    def compute(speaking_times: Dict[str, float],
                 total_duration: float) -> Dict[str, float]:
        if total_duration <= 0:
            return {k: 0.0 for k in speaking_times}
        return {
            k: round((v / total_duration) * 100, 2)
            for k, v in speaking_times.items()
        }


class SilentParticipantDetector:
    """Module 5: classify Silent / Moderate / Active."""

    @staticmethod
    def classify(participation_pct: Dict[str, float]) -> Dict[str, str]:
        result = {}
        for p, pct in participation_pct.items():
            if pct < Config.SILENT_THRESHOLD_PCT:
                result[p] = "Silent"
            elif pct < Config.MODERATE_THRESHOLD_PCT:
                result[p] = "Moderate"
            else:
                result[p] = "Active"
        return result


class DominantParticipantDetector:
    """Module 6: detect dominant participants (overrides classification)."""

    @staticmethod
    def classify(participation_pct: Dict[str, float],
                  classifications: Dict[str, str]) -> Dict[str, str]:
        updated = dict(classifications)
        for p, pct in participation_pct.items():
            if pct >= Config.DOMINANT_THRESHOLD_PCT:
                updated[p] = "Dominant"
        return updated


def run_phase2(transcript: List[TranscriptSegment],
               total_duration: float
               ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    speaking_times = SpeakingTimeAnalyzer.compute(transcript)
    participation_pct = ParticipationAnalyzer.compute(speaking_times, total_duration)
    classifications = SilentParticipantDetector.classify(participation_pct)
    classifications = DominantParticipantDetector.classify(participation_pct, classifications)
    return speaking_times, participation_pct, classifications


# ===================================================================
# PHASE 3 - CONVERSATION DYNAMICS
# ===================================================================

class InterruptionDetector:
    """
    Module 7: counts interruptions.

    An interruption occurs when participant B starts speaking
    before participant A's segment has ended (B's start < A's end),
    and B's resulting overlapping/leading segment lasts at least
    MIN_INTERRUPTING_SEGMENT_SEC, and A continues speaking afterward
    or A's turn is cut short (A does not voluntarily finish - i.e.
    A's end is pushed earlier than it otherwise would be, approximated
    by B starting strictly before A ends with sufficient overlap).
    """

    @staticmethod
    def compute(transcript: List[TranscriptSegment]
                 ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Dict[str, int]]]:
        made: Dict[str, int] = defaultdict(int)
        received: Dict[str, int] = defaultdict(int)
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        sorted_segs = sorted(transcript, key=lambda s: s.start)

        for i in range(len(sorted_segs) - 1):
            current = sorted_segs[i]
            for j in range(i + 1, len(sorted_segs)):
                nxt = sorted_segs[j]
                if nxt.speaker == current.speaker:
                    continue
                if nxt.start >= current.end:
                    break  # no more candidates start before current ends

                overlap = min(current.end, nxt.end) - max(current.start, nxt.start)
                starts_before_end = nxt.start < (current.end - Config.INTERRUPTION_GAP_SEC)

                if starts_before_end and overlap >= Config.MIN_INTERRUPTING_SEGMENT_SEC:
                    made[nxt.speaker] += 1
                    received[current.speaker] += 1
                    matrix[nxt.speaker][current.speaker] += 1
                break  # only consider the immediate next overlapping speaker

        return dict(made), dict(received), {k: dict(v) for k, v in matrix.items()}


class OverlapDetector:
    """Module 8: detect simultaneous speech."""

    @staticmethod
    def compute(transcript: List[TranscriptSegment]
                 ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        overlaps = []
        per_speaker_overlap: Dict[str, float] = defaultdict(float)

        sorted_segs = sorted(transcript, key=lambda s: s.start)
        for i in range(len(sorted_segs)):
            for j in range(i + 1, len(sorted_segs)):
                a, b = sorted_segs[i], sorted_segs[j]
                if b.start >= a.end:
                    break
                if a.speaker == b.speaker:
                    continue
                ov_start = max(a.start, b.start)
                ov_end = min(a.end, b.end)
                ov_dur = ov_end - ov_start
                if ov_dur >= Config.MIN_OVERLAP_SEC:
                    overlaps.append({
                        "speaker_a": a.speaker,
                        "speaker_b": b.speaker,
                        "start": round(ov_start, 2),
                        "end": round(ov_end, 2),
                        "duration": round(ov_dur, 2),
                    })
                    per_speaker_overlap[a.speaker] += ov_dur
                    per_speaker_overlap[b.speaker] += ov_dur

        return overlaps, dict(per_speaker_overlap)


class TurnTakingAnalyzer:
    """
    Module 9: turn-taking analysis.

    Healthy pattern: A -> B -> C -> D (varied speaker transitions)
    Unhealthy pattern: A -> A -> A -> A (repeated same speaker)

    Score (0-100) rewards:
      - high speaker-transition diversity (unique bigrams of consecutive
        speakers vs total transitions)
      - low rate of "self-repeats" (same speaker speaking again
        immediately after themselves)
      - even distribution of how often each participant takes a turn
    """

    @staticmethod
    def compute(transcript: List[TranscriptSegment]
                 ) -> Tuple[float, List[str]]:
        sorted_segs = sorted(transcript, key=lambda s: s.start)

        # Merge consecutive same-speaker segments into a single "turn"
        turn_sequence: List[str] = []
        for seg in sorted_segs:
            if not turn_sequence or turn_sequence[-1] != seg.speaker:
                turn_sequence.append(seg.speaker)

        if len(turn_sequence) < 2:
            return 0.0, turn_sequence

        transitions = list(zip(turn_sequence[:-1], turn_sequence[1:]))
        total_transitions = len(transitions)

        # Self-repeats should already be removed by construction, so
        # instead measure "ping-pong" (A->B->A->B...) vs broader rotation
        unique_pairs = len(set(transitions))
        transition_diversity = unique_pairs / total_transitions

        # Turn frequency balance: how evenly speakers take turns
        turn_counts = Counter(turn_sequence)
        n_speakers = len(turn_counts)
        if n_speakers > 1:
            ideal = len(turn_sequence) / n_speakers
            variance = sum((c - ideal) ** 2 for c in turn_counts.values()) / n_speakers
            max_variance = ideal ** 2 if ideal > 0 else 1
            balance_score = max(0.0, 1 - (variance / max_variance)) if max_variance > 0 else 1.0
        else:
            balance_score = 0.0

        # Combine: 50% transition diversity, 50% turn-frequency balance
        score = (0.5 * transition_diversity + 0.5 * balance_score) * 100
        return round(score, 2), turn_sequence


def run_phase3(transcript: List[TranscriptSegment]) -> Dict[str, Any]:
    made, received, matrix = InterruptionDetector.compute(transcript)
    overlaps, overlap_totals = OverlapDetector.compute(transcript)
    turn_score, turn_sequence = TurnTakingAnalyzer.compute(transcript)

    return {
        "interruptions_made": made,
        "interruptions_received": received,
        "interruption_matrix": matrix,
        "overlaps": overlaps,
        "overlap_totals": overlap_totals,
        "turn_taking_score": turn_score,
        "turn_sequence": turn_sequence,
    }


# ===================================================================
# PHASE 4 - COMMUNICATION QUALITY
# ===================================================================

class SpeakingPaceAnalyzer:
    """Module 10: words per minute."""

    @staticmethod
    def compute(transcript: List[TranscriptSegment]
                 ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, str]]:
        wpm: Dict[str, float] = {}
        word_counts: Dict[str, int] = defaultdict(int)
        speaking_time: Dict[str, float] = defaultdict(float)
        labels: Dict[str, str] = {}

        for seg in transcript:
            words = re.findall(r"[A-Za-z']+", seg.text)
            word_counts[seg.speaker] += len(words)
            speaking_time[seg.speaker] += seg.duration

        for speaker, wc in word_counts.items():
            duration_min = speaking_time[speaker] / 60.0
            rate = wc / duration_min if duration_min > 0 else 0.0
            wpm[speaker] = round(rate, 1)

            if rate < Config.PACE_SLOW_WPM:
                labels[speaker] = "Slow"
            elif rate > Config.PACE_FAST_WPM:
                labels[speaker] = "Fast"
            else:
                labels[speaker] = "Normal"

        return wpm, dict(word_counts), labels


class FillerWordDetector:
    """Module 11: filler word detection and density."""

    @staticmethod
    def compute(transcript: List[TranscriptSegment],
                 word_counts: Dict[str, int]
                 ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, Counter]]:
        filler_counts: Dict[str, int] = defaultdict(int)
        filler_breakdown: Dict[str, Counter] = defaultdict(Counter)

        # Sort multi-word fillers first so they match before single-word ones
        multi_word_fillers = sorted(
            (f for f in Config.FILLER_WORDS if " " in f),
            key=len, reverse=True
        )
        single_word_fillers = {f for f in Config.FILLER_WORDS if " " not in f}

        for seg in transcript:
            text_lower = seg.text.lower()

            remaining = text_lower
            for phrase in multi_word_fillers:
                count = remaining.count(phrase)
                if count:
                    filler_counts[seg.speaker] += count
                    filler_breakdown[seg.speaker][phrase] += count
                    remaining = remaining.replace(phrase, " ")

            tokens = re.findall(r"[a-z']+", remaining)
            for tok in tokens:
                if tok in single_word_fillers:
                    filler_counts[seg.speaker] += 1
                    filler_breakdown[seg.speaker][tok] += 1

        density: Dict[str, float] = {}
        for speaker, wc in word_counts.items():
            fc = filler_counts.get(speaker, 0)
            density[speaker] = round((fc / wc) * 100, 2) if wc > 0 else 0.0

        return dict(filler_counts), density, dict(filler_breakdown)


class PauseAnalyzer:
    """Module 12: pause / hesitation / thinking-gap analysis."""

    @staticmethod
    def compute(transcript: List[TranscriptSegment]
                 ) -> Dict[str, Dict[str, int]]:
        result: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"long_pauses": 0, "hesitation_pauses": 0})

        by_speaker: Dict[str, List[TranscriptSegment]] = defaultdict(list)
        for seg in transcript:
            by_speaker[seg.speaker].append(seg)

        for speaker, segs in by_speaker.items():
            segs_sorted = sorted(segs, key=lambda s: s.start)
            for i in range(len(segs_sorted) - 1):
                gap = segs_sorted[i + 1].start - segs_sorted[i].end
                if gap >= Config.LONG_PAUSE_SEC:
                    result[speaker]["long_pauses"] += 1
                elif gap >= Config.HESITATION_PAUSE_SEC:
                    result[speaker]["hesitation_pauses"] += 1

            # Also check intra-segment word gaps for hesitations
            for seg in segs_sorted:
                words = seg.words
                for i in range(len(words) - 1):
                    gap = words[i + 1]["start"] - words[i]["end"]
                    if gap >= Config.LONG_PAUSE_SEC:
                        result[speaker]["long_pauses"] += 1
                    elif gap >= Config.HESITATION_PAUSE_SEC:
                        result[speaker]["hesitation_pauses"] += 1

        return dict(result)


def run_phase4(transcript: List[TranscriptSegment]) -> Dict[str, Any]:
    wpm, word_counts, pace_labels = SpeakingPaceAnalyzer.compute(transcript)
    filler_counts, filler_density, filler_breakdown = FillerWordDetector.compute(
        transcript, word_counts)
    pauses = PauseAnalyzer.compute(transcript)

    return {
        "wpm": wpm,
        "word_counts": word_counts,
        "pace_labels": pace_labels,
        "filler_counts": filler_counts,
        "filler_density": filler_density,
        "filler_breakdown": filler_breakdown,
        "pauses": pauses,
    }


# ===================================================================
# PHASE 5 - NLP ANALYSIS
# ===================================================================

class KeywordExtractor:
    """Module 13: keyword extraction using spaCy + KeyBERT."""

    def __init__(self):
        self._nlp = None
        self._kw_model = None

    def _load_spacy(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def _load_keybert(self):
        if self._kw_model is None:
            from keybert import KeyBERT
            self._kw_model = KeyBERT()
        return self._kw_model

    def extract_per_speaker(self, transcript: List[TranscriptSegment],
                              top_n: int = 5) -> Dict[str, List[str]]:
        by_speaker: Dict[str, List[str]] = defaultdict(list)
        for seg in transcript:
            by_speaker[seg.speaker].append(seg.text)

        results: Dict[str, List[str]] = {}
        try:
            kw_model = self._load_keybert()
        except Exception:
            kw_model = None

        for speaker, texts in by_speaker.items():
            full_text = " ".join(texts)
            if len(full_text.split()) < 3:
                results[speaker] = []
                continue
            if kw_model is None:
                results[speaker] = self._fallback_keywords(full_text, top_n)
                continue
            try:
                keywords = kw_model.extract_keywords(
                    full_text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words="english",
                    top_n=top_n,
                )
                results[speaker] = [k[0] for k in keywords]
            except Exception:
                results[speaker] = self._fallback_keywords(full_text, top_n)

        return results

    def extract_global(self, transcript: List[TranscriptSegment],
                         top_n: int = 10) -> List[str]:
        full_text = " ".join(seg.text for seg in transcript)
        try:
            kw_model = self._load_keybert()
            keywords = kw_model.extract_keywords(
                full_text, keyphrase_ngram_range=(1, 2),
                stop_words="english", top_n=top_n)
            return [k[0] for k in keywords]
        except Exception:
            return self._fallback_keywords(full_text, top_n)

    def _fallback_keywords(self, text: str, top_n: int) -> List[str]:
        """Simple noun-based fallback if spaCy/KeyBERT unavailable."""
        try:
            nlp = self._load_spacy()
            doc = nlp(text)
            nouns = [tok.lemma_.lower() for tok in doc
                     if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop
                     and len(tok.text) > 2]
            counts = Counter(nouns)
            return [w for w, _ in counts.most_common(top_n)]
        except Exception:
            words = re.findall(r"[a-zA-Z]{3,}", text.lower())
            common_stop = {"the", "and", "for", "that", "this", "with",
                            "have", "are", "was", "were", "but"}
            filtered = [w for w in words if w not in common_stop]
            counts = Counter(filtered)
            return [w for w, _ in counts.most_common(top_n)]


class TopicTimelineBuilder:
    """Module 14: identify topic evolution over time windows."""

    def __init__(self, extractor: Optional[KeywordExtractor] = None):
        self.extractor = extractor or KeywordExtractor()

    def build(self, transcript: List[TranscriptSegment],
              window_sec: float = 300.0) -> List[Dict[str, Any]]:
        if not transcript:
            return []

        total_end = max(seg.end for seg in transcript)
        timeline = []

        window_start = 0.0
        while window_start < total_end:
            window_end = min(window_start + window_sec, total_end)
            window_segs = [
                seg for seg in transcript
                if seg.start < window_end and seg.end > window_start
            ]
            text = " ".join(seg.text for seg in window_segs)
            if text.strip():
                topics = self.extractor._fallback_keywords(text, top_n=3) \
                    if not text.strip() else self.extractor.extract_global(
                        [TranscriptSegment("_", 0, 0, text)], top_n=3)
            else:
                topics = []

            timeline.append({
                "start_min": round(window_start / 60, 1),
                "end_min": round(window_end / 60, 1),
                "topics": topics,
            })
            window_start = window_end

        return timeline


class TopicDriftDetector:
    """
    Module 15: detect off-topic conversations by measuring keyword
    overlap (Jaccard similarity) between each window and the overall
    session topic set. Low overlap => potential drift.
    """

    @staticmethod
    def detect(topic_timeline: List[Dict[str, Any]],
               global_keywords: List[str]) -> List[Dict[str, Any]]:
        drift_events = []
        global_set = set(k.lower() for k in global_keywords)
        if not global_set:
            return drift_events

        for window in topic_timeline:
            window_set = set(t.lower() for t in window["topics"])
            if not window_set:
                continue
            intersection = window_set & global_set
            union = window_set | global_set
            similarity = len(intersection) / len(union) if union else 0.0

            if similarity < Config.TOPIC_SIMILARITY_THRESHOLD:
                drift_events.append({
                    "start_min": window["start_min"],
                    "end_min": window["end_min"],
                    "window_topics": window["topics"],
                    "similarity_to_global": round(similarity, 2),
                    "flag": "Possible topic drift",
                })

        return drift_events


def run_phase5(transcript: List[TranscriptSegment]) -> Dict[str, Any]:
    extractor = KeywordExtractor()
    per_speaker_keywords = extractor.extract_per_speaker(transcript)
    global_keywords = extractor.extract_global(transcript)

    timeline_builder = TopicTimelineBuilder(extractor)
    topic_timeline = timeline_builder.build(transcript)

    drift_events = TopicDriftDetector.detect(topic_timeline, global_keywords)

    return {
        "per_speaker_keywords": per_speaker_keywords,
        "global_keywords": global_keywords,
        "topic_timeline": topic_timeline,
        "topic_drift_events": drift_events,
    }


# ===================================================================
# PHASE 6 - INTERACTION GRAPH
# ===================================================================

class ResponseGraphBuilder:
    """
    Module 16: builds a directed graph A -> B meaning "B spoke
    immediately after A" (a response/turn relationship).
    """

    @staticmethod
    def build(transcript: List[TranscriptSegment]) -> nx.DiGraph:
        sorted_segs = sorted(transcript, key=lambda s: s.start)
        G = nx.DiGraph()

        for seg in sorted_segs:
            G.add_node(seg.speaker)

        for i in range(len(sorted_segs) - 1):
            a = sorted_segs[i].speaker
            b = sorted_segs[i + 1].speaker
            if a == b:
                continue
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

        return G


class InteractionFrequencyAnalyzer:
    """Module 17: who responds to whom, as a weighted edge list."""

    @staticmethod
    def compute(graph: nx.DiGraph) -> List[Tuple[str, str, int]]:
        return [(u, v, d["weight"]) for u, v, d in graph.edges(data=True)]


class ParticipationBalanceScorer:
    """
    Module 18: participation balance score (0-100).

    Based on the normalized entropy of the participation-percentage
    distribution (Shannon evenness). 100 = perfectly equal
    participation across all participants; lower = more skewed.
    """

    @staticmethod
    def compute(participation_pct: Dict[str, float]) -> float:
        if not participation_pct:
            return 0.0

        proportions = [p / 100.0 for p in participation_pct.values() if p > 0]
        if not proportions:
            return 0.0

        n = len(proportions)
        if n <= 1:
            return 100.0 if n == 1 else 0.0

        entropy = -sum(p * np.log(p) for p in proportions)
        max_entropy = np.log(n)

        evenness = entropy / max_entropy if max_entropy > 0 else 0.0
        return round(evenness * 100, 2)


def run_phase6(transcript: List[TranscriptSegment],
               participation_pct: Dict[str, float]
               ) -> Dict[str, Any]:
    graph = ResponseGraphBuilder.build(transcript)
    edges = InteractionFrequencyAnalyzer.compute(graph)
    balance_score = ParticipationBalanceScorer.compute(participation_pct)

    return {
        "response_graph": graph,
        "response_graph_edges": edges,
        "participation_balance_score": balance_score,
    }


# ===================================================================
# PHASE 7 - GROUP METRICS
# ===================================================================

class CommunicationScorer:
    """
    Module 19: per-participant communication score (0-100).

    Weighted combination of:
      - participation balance contribution (closeness to ideal share)
      - pace appropriateness (Normal pace scores highest)
      - filler density penalty
      - interruption penalty (made + received)
    """

    @staticmethod
    def compute(participant_id: str,
                participation_pct: float,
                ideal_pct: float,
                pace_label: str,
                filler_density: float,
                interruptions_made: int,
                interruptions_received: int) -> float:

        # Participation closeness (penalize both too little and too much)
        deviation = abs(participation_pct - ideal_pct)
        participation_score = max(0.0, 100 - (deviation / ideal_pct) * 100) if ideal_pct > 0 else 50.0
        participation_score = min(participation_score, 100.0)

        # Pace score
        pace_score = {"Normal": 100.0, "Slow": 70.0, "Fast": 65.0}.get(pace_label, 70.0)

        # Filler penalty: 0% filler -> 100, 20%+ -> 0
        filler_score = max(0.0, 100 - (filler_density / 20.0) * 100)

        # Interruption penalty: each interruption made costs 5 pts,
        # each received costs 2 pts
        interruption_penalty = (interruptions_made * 5) + (interruptions_received * 2)
        interruption_score = max(0.0, 100 - interruption_penalty)

        final_score = (
            0.35 * participation_score +
            0.20 * pace_score +
            0.20 * filler_score +
            0.25 * interruption_score
        )
        return round(final_score, 2)


class GroupCohesionScorer:
    """
    Module 20: group cohesion score (0-100) and category.

    Based on:
      - participation balance score (Phase 6)
      - turn-taking score (Phase 3)
      - average communication score
      - total overlap time relative to session duration (penalty)
    """

    @staticmethod
    def compute(participation_balance: float,
                turn_taking_score: float,
                avg_communication_score: float,
                total_overlap_sec: float,
                total_duration_sec: float) -> Tuple[float, str]:

        overlap_ratio = (total_overlap_sec / total_duration_sec) if total_duration_sec > 0 else 0
        overlap_penalty = min(overlap_ratio * 200, 30)  # cap penalty at 30 pts

        raw_score = (
            0.35 * participation_balance +
            0.30 * turn_taking_score +
            0.35 * avg_communication_score
        ) - overlap_penalty

        score = round(max(0.0, min(100.0, raw_score)), 2)

        if score >= 85:
            label = "Excellent"
        elif score >= 70:
            label = "Good"
        elif score >= 50:
            label = "Average"
        else:
            label = "Poor"

        return score, label


def run_phase7(participants: Dict[str, ParticipantMetrics],
               participation_balance: float,
               turn_taking_score: float,
               overlap_totals: Dict[str, float],
               total_duration_sec: float) -> Tuple[float, str]:

    n = len(participants)
    ideal_pct = 100.0 / n if n > 0 else 0.0

    for pid, pm in participants.items():
        pm.communication_score = CommunicationScorer.compute(
            participant_id=pid,
            participation_pct=pm.participation_pct,
            ideal_pct=ideal_pct,
            pace_label=pm.pace_label,
            filler_density=pm.filler_density_pct,
            interruptions_made=pm.interruptions_made,
            interruptions_received=pm.interruptions_received,
        )

    avg_comm_score = (
        sum(p.communication_score for p in participants.values()) / n
        if n > 0 else 0.0
    )

    total_overlap = sum(overlap_totals.values())

    cohesion_score, cohesion_label = GroupCohesionScorer.compute(
        participation_balance=participation_balance,
        turn_taking_score=turn_taking_score,
        avg_communication_score=avg_comm_score,
        total_overlap_sec=total_overlap,
        total_duration_sec=total_duration_sec,
    )

    return cohesion_score, cohesion_label


# ===================================================================
# PHASE 8 - VISUALIZATION DASHBOARD
# ===================================================================

class DashboardBuilder:
    """
    Phase 8: builds Plotly figures for participation, speaking time,
    timeline, and interaction network, plus a Matplotlib summary.
    """

    @staticmethod
    def build_participation_pie(participation_pct: Dict[str, float]):
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Pie(
            labels=list(participation_pct.keys()),
            values=list(participation_pct.values()),
            hole=0.35,
        )])
        fig.update_layout(title="Participation Percentage")
        return fig

    @staticmethod
    def build_speaking_time_bar(speaking_times: Dict[str, float]):
        import plotly.graph_objects as go
        labels = list(speaking_times.keys())
        values = [round(v, 1) for v in speaking_times.values()]
        fig = go.Figure(data=[go.Bar(x=labels, y=values)])
        fig.update_layout(
            title="Speaking Time (seconds)",
            xaxis_title="Participant",
            yaxis_title="Seconds",
        )
        return fig

    @staticmethod
    def build_timeline_chart(transcript: List[TranscriptSegment]):
        import plotly.express as px
        df = pd.DataFrame([
            {
                "Speaker": seg.speaker,
                "Start": seg.start,
                "Finish": seg.end,
            }
            for seg in transcript
        ])
        if df.empty:
            return None

        # px.timeline requires datetime; convert seconds to timestamps
        epoch = pd.Timestamp("2024-01-01")
        df["Start_dt"] = epoch + pd.to_timedelta(df["Start"], unit="s")
        df["Finish_dt"] = epoch + pd.to_timedelta(df["Finish"], unit="s")

        fig = px.timeline(
            df, x_start="Start_dt", x_end="Finish_dt", y="Speaker",
            color="Speaker", title="Discussion Timeline"
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    @staticmethod
    def build_network_graph(graph: nx.DiGraph):
        import plotly.graph_objects as go

        if graph.number_of_nodes() == 0:
            return None

        pos = nx.spring_layout(graph, seed=42)

        edge_x, edge_y = [], []
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1, color="#888"), hoverinfo="none"
        )

        node_x = [pos[n][0] for n in graph.nodes()]
        node_y = [pos[n][1] for n in graph.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=list(graph.nodes()), textposition="top center",
            marker=dict(size=20, color="#1f77b4"),
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Interaction Network (who speaks after whom)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        return fig

    @staticmethod
    def build_matplotlib_summary(participants: Dict[str, ParticipantMetrics],
                                  output_path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ids = list(participants.keys())
        comm_scores = [participants[p].communication_score for p in ids]
        participation = [participants[p].participation_pct for p in ids]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].bar(ids, participation, color="#4C72B0")
        axes[0].set_title("Participation %")
        axes[0].set_ylabel("Percent")
        axes[0].tick_params(axis="x", rotation=30)

        axes[1].bar(ids, comm_scores, color="#55A868")
        axes[1].set_title("Communication Score")
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        return output_path

    @staticmethod
    def save_all(result: "SessionResult", graph: nx.DiGraph,
                  output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        speaking_times = {pid: pm.speaking_time_sec
                            for pid, pm in result.participants.items()}
        participation_pct = {pid: pm.participation_pct
                               for pid, pm in result.participants.items()}

        pie = DashboardBuilder.build_participation_pie(participation_pct)
        pie_path = os.path.join(output_dir, "participation_pie.html")
        pie.write_html(pie_path)
        paths["participation_pie"] = pie_path

        bar = DashboardBuilder.build_speaking_time_bar(speaking_times)
        bar_path = os.path.join(output_dir, "speaking_time_bar.html")
        bar.write_html(bar_path)
        paths["speaking_time_bar"] = bar_path

        timeline = DashboardBuilder.build_timeline_chart(result.transcript)
        if timeline is not None:
            timeline_path = os.path.join(output_dir, "timeline.html")
            timeline.write_html(timeline_path)
            paths["timeline"] = timeline_path

        network = DashboardBuilder.build_network_graph(graph)
        if network is not None:
            network_path = os.path.join(output_dir, "network_graph.html")
            network.write_html(network_path)
            paths["network_graph"] = network_path

        mpl_path = os.path.join(output_dir, "summary.png")
        DashboardBuilder.build_matplotlib_summary(result.participants, mpl_path)
        paths["matplotlib_summary"] = mpl_path

        return paths


def run_phase8(result: "SessionResult", graph: nx.DiGraph,
               output_dir: str = Config.OUTPUT_DIR) -> Dict[str, str]:
    return DashboardBuilder.save_all(result, graph, output_dir)


# ===================================================================
# PHASE 9 - PDF REPORT GENERATION
# ===================================================================

class PDFReportGenerator:
    """Generates a participant-wise session PDF report using ReportLab."""

    @staticmethod
    def generate(result: "SessionResult", output_path: str) -> str:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=12)
        heading_style = ParagraphStyle(
            "HeadingCustom", parent=styles["Heading2"], spaceAfter=8)
        normal_style = styles["Normal"]

        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)

        elements = []

        # --- Cover / session summary ---
        elements.append(Paragraph("EdgeGD Session Report", title_style))
        elements.append(Paragraph(f"Session ID: {result.session_id}", normal_style))
        elements.append(Paragraph(f"Date: {result.date}", normal_style))
        elements.append(Paragraph(f"Topic: {result.topic}", normal_style))
        elements.append(Paragraph(
            f"Total Duration: {result.total_duration_sec/60:.1f} min", normal_style))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Group Metrics", heading_style))
        group_data = [
            ["Metric", "Value"],
            ["Participation Balance Score", f"{result.participation_balance_score} / 100"],
            ["Turn-Taking Score", f"{result.turn_taking_score} / 100"],
            ["Group Cohesion Score", f"{result.group_cohesion_score} / 100"],
            ["Group Cohesion Category", result.group_cohesion_label],
        ]
        group_table = Table(group_data, colWidths=[8*cm, 8*cm])
        group_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(group_table)
        elements.append(Spacer(1, 12))

        # --- Topic timeline ---
        if result.topic_timeline:
            elements.append(Paragraph("Topic Timeline", heading_style))
            timeline_data = [["Window (min)", "Topics"]]
            for w in result.topic_timeline:
                timeline_data.append([
                    f"{w['start_min']} - {w['end_min']}",
                    ", ".join(w["topics"]) if w["topics"] else "-"
                ])
            timeline_table = Table(timeline_data, colWidths=[4*cm, 12*cm])
            timeline_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]))
            elements.append(timeline_table)
            elements.append(Spacer(1, 12))

        if result.topic_drift_events:
            elements.append(Paragraph("Topic Drift Alerts", heading_style))
            for ev in result.topic_drift_events:
                elements.append(Paragraph(
                    f"- {ev['start_min']}-{ev['end_min']} min: "
                    f"{ev['flag']} (similarity={ev['similarity_to_global']})",
                    normal_style))
            elements.append(Spacer(1, 12))

        elements.append(PageBreak())

        # --- Participant-wise reports ---
        elements.append(Paragraph("Participant-Wise Reports", title_style))

        for pid, pm in result.participants.items():
            elements.append(Paragraph(f"Participant: {pid}", heading_style))

            metrics_data = [
                ["Metric", "Value"],
                ["Speaking Time", f"{pm.speaking_time_sec/60:.2f} min"],
                ["Participation", f"{pm.participation_pct}%"],
                ["Classification", pm.classification],
                ["Interruptions Made", str(pm.interruptions_made)],
                ["Interruptions Received", str(pm.interruptions_received)],
                ["Speaking Pace", f"{pm.avg_wpm} WPM ({pm.pace_label})"],
                ["Filler Density", f"{pm.filler_density_pct}%"],
                ["Long Pauses", str(pm.long_pauses)],
                ["Hesitation Pauses", str(pm.hesitation_pauses)],
                ["Keywords", ", ".join(pm.keywords) if pm.keywords else "-"],
                ["Communication Score", f"{pm.communication_score} / 100"],
            ]
            t = Table(metrics_data, colWidths=[6*cm, 10*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16A085")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 6))

            if pm.strengths:
                elements.append(Paragraph(
                    "<b>Strengths:</b> " + "; ".join(pm.strengths), normal_style))
            if pm.weaknesses:
                elements.append(Paragraph(
                    "<b>Weaknesses:</b> " + "; ".join(pm.weaknesses), normal_style))

            elements.append(Spacer(1, 14))

        doc.build(elements)
        return output_path


def run_phase9(result: "SessionResult",
               output_dir: str = Config.OUTPUT_DIR) -> str:
    output_path = os.path.join(output_dir, f"report_{result.session_id}.pdf")
    return PDFReportGenerator.generate(result, output_path)


# ===================================================================
# STRENGTHS / WEAKNESSES GENERATION (feeds Phase 9 report)
# ===================================================================

class FeedbackGenerator:
    """Generates explainable strengths/weaknesses per participant."""

    @staticmethod
    def generate(pm: ParticipantMetrics, n_participants: int) -> Tuple[List[str], List[str]]:
        strengths, weaknesses = [], []
        ideal_pct = 100.0 / n_participants if n_participants > 0 else 0

        if pm.classification == "Active":
            strengths.append("Balanced, active participation in the discussion")
        elif pm.classification == "Dominant":
            weaknesses.append("Dominated discussion time, limiting others' contributions")
        elif pm.classification == "Moderate":
            strengths.append("Contributed steadily without overtaking the discussion")
        elif pm.classification == "Silent":
            weaknesses.append("Very limited verbal participation; consider encouraging more input")

        if pm.pace_label == "Normal":
            strengths.append("Maintained a clear, easy-to-follow speaking pace")
        elif pm.pace_label == "Slow":
            weaknesses.append("Speaking pace was slower than typical conversational range")
        elif pm.pace_label == "Fast":
            weaknesses.append("Speaking pace was faster than ideal, which may reduce clarity")

        if pm.filler_density_pct <= 3:
            strengths.append("Minimal use of filler words; concise communication")
        elif pm.filler_density_pct <= 8:
            weaknesses.append("Moderate use of filler words ('um', 'like', etc.)")
        else:
            weaknesses.append("High filler word density, which may reduce perceived confidence")

        if pm.interruptions_made == 0:
            strengths.append("Did not interrupt other participants")
        elif pm.interruptions_made <= 2:
            weaknesses.append("Occasionally interrupted other participants")
        else:
            weaknesses.append("Frequently interrupted other participants")

        if pm.interruptions_received > 2:
            weaknesses.append("Was interrupted frequently by others")

        if pm.long_pauses > 3:
            weaknesses.append("Several long pauses, suggesting hesitation or uncertainty")

        return strengths, weaknesses


# ===================================================================
# DATABASE LAYER (PostgreSQL via SQLAlchemy, per documentation schema)
# ===================================================================

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()


class ParticipantDB(Base):
    __tablename__ = "participants"
    participant_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    voice_embedding = Column(JSON, nullable=True)


class SessionDB(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    date = Column(String)
    topic = Column(String)
    duration = Column(Float)

    metrics = relationship("MetricsDB", back_populates="session")


class MetricsDB(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    participant_id = Column(String)
    speaking_time = Column(Float)
    participation_percentage = Column(Float)
    interruptions = Column(Integer)
    pace = Column(String)
    filler_density = Column(Float)
    communication_score = Column(Float)

    session = relationship("SessionDB", back_populates="metrics")


class DatabaseManager:
    def __init__(self, db_url: str = Config.DB_URL):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_session(self, result: "SessionResult"):
        db = self.Session()
        try:
            db.merge(SessionDB(
                session_id=result.session_id,
                date=result.date,
                topic=result.topic,
                duration=result.total_duration_sec,
            ))

            for pid, pm in result.participants.items():
                db.merge(ParticipantDB(participant_id=pid, name=pid))
                db.add(MetricsDB(
                    session_id=result.session_id,
                    participant_id=pid,
                    speaking_time=pm.speaking_time_sec,
                    participation_percentage=pm.participation_pct,
                    interruptions=pm.interruptions_made,
                    pace=pm.pace_label,
                    filler_density=pm.filler_density_pct,
                    communication_score=pm.communication_score,
                ))
            db.commit()
        finally:
            db.close()


# ===================================================================
# PHASE 1-9 ORCHESTRATION (batch / offline pipeline)
# ===================================================================

def analyze_session(audio_path: str,
                     topic: str = "Group Discussion",
                     num_speakers: Optional[int] = None,
                     output_dir: str = Config.OUTPUT_DIR,
                     save_to_db: bool = False,
                     progress: Optional[Any] = None,
                     stt: Optional[SpeechToTextEngine] = None,
                     diarizer: Optional[SpeakerDiarizer] = None,
                     session_id: Optional[str] = None,
                     total_duration_sec: Optional[float] = None) -> "SessionResult":
    """
    Full offline pipeline: Phases 1-9.

    `progress` is an optional callable(label, pct) so a caller (e.g. the web
    backend) can report which phase is running. `stt` / `diarizer` allow
    reusing warm models. `total_duration_sec`, when given, is used as the
    session length instead of the last spoken word's timestamp - the real
    recording length is what participation percentages should divide by.
    """
    session_id = session_id or str(uuid.uuid4())[:8]
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    def tick(label: str, pct: int):
        if progress:
            progress(label, pct)

    # --- Phase 1 ---
    transcript = run_phase1(audio_path, num_speakers=num_speakers,
                            stt=stt, diarizer=diarizer,
                            progress=lambda l, p: tick(l, int(p * 0.62)))
    spoken_end = max((seg.end for seg in transcript), default=0.0)
    total_duration = total_duration_sec or spoken_end

    # --- Phase 2 ---
    tick("Measuring participation", 64)
    speaking_times, participation_pct, classifications = run_phase2(transcript, total_duration)

    # --- Phase 3 ---
    tick("Analysing conversation dynamics", 68)
    p3 = run_phase3(transcript)

    # --- Phase 4 ---
    tick("Scoring communication quality", 72)
    p4 = run_phase4(transcript)

    # --- Phase 5 ---
    tick("Extracting keywords and topics", 76)
    p5 = run_phase5(transcript)

    # --- Phase 6 ---
    tick("Building interaction graph", 84)
    p6 = run_phase6(transcript, participation_pct)

    # --- Assemble participant metrics ---
    participants: Dict[str, ParticipantMetrics] = {}
    all_speakers = set(speaking_times.keys()) | set(participation_pct.keys())

    for pid in all_speakers:
        pm = ParticipantMetrics(participant_id=pid)
        pm.speaking_time_sec = speaking_times.get(pid, 0.0)
        pm.participation_pct = participation_pct.get(pid, 0.0)
        pm.classification = classifications.get(pid, "Unknown")
        pm.interruptions_made = p3["interruptions_made"].get(pid, 0)
        pm.interruptions_received = p3["interruptions_received"].get(pid, 0)
        pm.overlap_time_sec = p3["overlap_totals"].get(pid, 0.0)
        pm.word_count = p4["word_counts"].get(pid, 0)
        pm.avg_wpm = p4["wpm"].get(pid, 0.0)
        pm.pace_label = p4["pace_labels"].get(pid, "Normal")
        pm.filler_count = p4["filler_counts"].get(pid, 0)
        pm.filler_density_pct = p4["filler_density"].get(pid, 0.0)
        pm.long_pauses = p4["pauses"].get(pid, {}).get("long_pauses", 0)
        pm.hesitation_pauses = p4["pauses"].get(pid, {}).get("hesitation_pauses", 0)
        pm.keywords = p5["per_speaker_keywords"].get(pid, [])

        participants[pid] = pm

    # --- Phase 7 ---
    tick("Scoring group cohesion", 88)
    cohesion_score, cohesion_label = run_phase7(
        participants=participants,
        participation_balance=p6["participation_balance_score"],
        turn_taking_score=p3["turn_taking_score"],
        overlap_totals=p3["overlap_totals"],
        total_duration_sec=total_duration,
    )

    # --- Strengths / weaknesses (for Phase 9) ---
    n_participants = len(participants)
    for pm in participants.values():
        strengths, weaknesses = FeedbackGenerator.generate(pm, n_participants)
        pm.strengths = strengths
        pm.weaknesses = weaknesses

    result = SessionResult(
        session_id=session_id,
        date=date_str,
        topic=topic,
        total_duration_sec=total_duration,
        transcript=transcript,
        participants=participants,
        interruption_matrix=p3["interruption_matrix"],
        overlaps=p3["overlaps"],
        turn_taking_score=p3["turn_taking_score"],
        turn_sequence=p3["turn_sequence"],
        topic_timeline=p5["topic_timeline"],
        topic_drift_events=p5["topic_drift_events"],
        response_graph_edges=p6["response_graph_edges"],
        participation_balance_score=p6["participation_balance_score"],
        group_cohesion_score=cohesion_score,
        group_cohesion_label=cohesion_label,
    )

    # --- Phase 8 ---
    tick("Rendering charts", 91)
    dashboard_paths = run_phase8(result, p6["response_graph"], output_dir)

    # --- Phase 9 ---
    tick("Writing PDF report", 96)
    pdf_path = run_phase9(result, output_dir)

    # --- Save JSON summary ---
    tick("Saving results", 99)
    json_path = os.path.join(output_dir, f"result_{session_id}.json")
    with open(json_path, "w") as f:
        json.dump(_result_to_dict(result), f, indent=2, default=str)

    if save_to_db:
        DatabaseManager().save_session(result)

    print(f"\nSession {session_id} complete.")
    print(f"  PDF report : {pdf_path}")
    print(f"  Dashboards : {dashboard_paths}")
    print(f"  JSON summary: {json_path}")

    return result


def _result_to_dict(result: "SessionResult") -> Dict[str, Any]:
    d = asdict(result)
    # transcript segments include raw word dicts; keep but trim for brevity
    d["transcript"] = [
        {"speaker": s["speaker"], "start": s["start"], "end": s["end"], "text": s["text"]}
        for s in d["transcript"]
    ]
    return d


# ===================================================================
# PHASE 10 - REAL-TIME EXTENSION
# ===================================================================

class LiveMetricsEngine:
    """
    Module: Live Metrics Engine (Phase 10).

    Consumes a rolling buffer of (speaker, start, end, text) segments
    produced from streaming audio and maintains live aggregate
    metrics: participation %, dominance/silence alerts, active
    speaker, and a simple live turn-taking visualization feed.
    """

    def __init__(self, buffer_window_sec: float = Config.LIVE_BUFFER_SEC):
        self.buffer_window_sec = buffer_window_sec
        self.segments: List[TranscriptSegment] = []
        self.start_time = time.time()
        self.lock = threading.Lock()

    def add_segment(self, seg: TranscriptSegment):
        with self.lock:
            self.segments.append(seg)
            # Drop segments outside the rolling window
            cutoff = seg.end - self.buffer_window_sec
            self.segments = [s for s in self.segments if s.end >= cutoff]

    def current_metrics(self) -> Dict[str, Any]:
        with self.lock:
            if not self.segments:
                return {"participants": {}, "active_speaker": None, "alerts": []}

            window_start = min(s.start for s in self.segments)
            window_end = max(s.end for s in self.segments)
            window_duration = max(window_end - window_start, 1e-6)

            speaking_times = SpeakingTimeAnalyzer.compute(self.segments)
            participation_pct = ParticipationAnalyzer.compute(speaking_times, window_duration)
            classifications = SilentParticipantDetector.classify(participation_pct)
            classifications = DominantParticipantDetector.classify(participation_pct, classifications)

            active_speaker = max(self.segments, key=lambda s: s.end).speaker

            alerts = []
            for pid, label in classifications.items():
                if label == "Dominant":
                    alerts.append(f"Dominance alert: {pid} ({participation_pct[pid]}%)")
                elif label == "Silent":
                    alerts.append(f"Silent participant alert: {pid} ({participation_pct[pid]}%)")

            _, turn_sequence = TurnTakingAnalyzer.compute(self.segments)

            return {
                "window_start": round(window_start, 1),
                "window_end": round(window_end, 1),
                "participation_pct": participation_pct,
                "classifications": classifications,
                "active_speaker": active_speaker,
                "alerts": alerts,
                "turn_sequence": turn_sequence[-10:],  # last 10 turns
            }


class StreamingPipeline:
    """
    Phase 10 pipeline:

      Microphone -> VAD -> Streaming Buffer -> Whisper ->
      Speaker Diarization (per-chunk) -> LiveMetricsEngine -> Dashboard

    Implementation notes:
      - Uses `sounddevice` to capture microphone audio.
      - A lightweight energy-based VAD gates silent chunks.
      - Each LIVE_CHUNK_SEC chunk is transcribed with Whisper.
      - Speaker labeling within a live chunk uses pyannote if
        available; otherwise falls back to a single "Speaker_00"
        label (diarization across short chunks is unreliable without
        speaker enrollment / embeddings).
    """

    def __init__(self, chunk_sec: float = Config.LIVE_CHUNK_SEC,
                 sample_rate: int = Config.SAMPLE_RATE,
                 num_speakers: Optional[int] = None):
        self.chunk_sec = chunk_sec
        self.sample_rate = sample_rate
        self.num_speakers = num_speakers

        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.metrics_engine = LiveMetricsEngine()

        self.stt = SpeechToTextEngine(model_size="tiny")  # smaller for latency
        self._diarizer = None  # lazy-loaded

        self._stop_event = threading.Event()

    # ---- Voice Activity Detection (simple energy-based) ----
    @staticmethod
    def _is_speech(audio_chunk: np.ndarray, threshold: float = 0.01) -> bool:
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
        return rms > threshold

    def _audio_callback(self, indata, frames, time_info, status):
        self.audio_queue.put(indata.copy())

    def _processing_loop(self):
        import tempfile
        import soundfile as sf

        buffer = np.zeros((0, 1), dtype=np.float32)
        chunk_samples = int(self.chunk_sec * self.sample_rate)

        while not self._stop_event.is_set():
            try:
                data = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, data], axis=0)

            while len(buffer) >= chunk_samples:
                chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                if not self._is_speech(chunk):
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, chunk, self.sample_rate)
                    tmp_path = tmp.name

                try:
                    whisper_result = self.stt.transcribe(tmp_path)
                    text = whisper_result.get("text", "").strip()
                    if not text:
                        continue

                    speaker = self._diarize_chunk(tmp_path)

                    now = time.time() - self.metrics_engine.start_time
                    seg = TranscriptSegment(
                        speaker=speaker,
                        start=now - self.chunk_sec,
                        end=now,
                        text=text,
                    )
                    self.metrics_engine.add_segment(seg)

                    metrics = self.metrics_engine.current_metrics()
                    self._render_dashboard(seg, metrics)

                finally:
                    os.remove(tmp_path)

    def _diarize_chunk(self, audio_path: str) -> str:
        """
        Best-effort per-chunk diarization. Loads pyannote lazily;
        falls back to a single speaker label if unavailable
        (recommended: integrate persistent speaker embeddings /
        enrollment for reliable real-time diarization).
        """
        try:
            if self._diarizer is None:
                self._diarizer = SpeakerDiarizer()
            segments = self._diarizer.diarize(audio_path, num_speakers=self.num_speakers)
            if segments:
                # Pick the speaker with the most coverage in this chunk
                durations: Dict[str, float] = defaultdict(float)
                for s in segments:
                    durations[s["speaker"]] += (s["end"] - s["start"])
                return max(durations, key=durations.get)
        except Exception:
            pass
        return "Speaker_00"

    def _render_dashboard(self, seg: TranscriptSegment, metrics: Dict[str, Any]):
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 60)
        print(" EdgeGD - LIVE SESSION DASHBOARD")
        print("=" * 60)
        print(f"\nLatest [{seg.speaker}] ({seg.start:.1f}-{seg.end:.1f}s):")
        print(f"  \"{seg.text}\"")

        print("\nLive Participation:")
        for pid, pct in metrics.get("participation_pct", {}).items():
            label = metrics["classifications"].get(pid, "")
            bar = "#" * int(pct / 2)
            print(f"  {pid:<14} {pct:5.1f}%  [{label:<8}] {bar}")

        if metrics.get("alerts"):
            print("\nALERTS:")
            for a in metrics["alerts"]:
                print(f"  ! {a}")

        print(f"\nActive speaker: {metrics.get('active_speaker')}")
        print(f"Recent turn sequence: {' -> '.join(metrics.get('turn_sequence', []))}")
        print("\n(Press Ctrl+C to stop)")

    def run(self, duration: Optional[float] = None):
        import sounddevice as sd

        print("Starting EdgeGD live session (Phase 10)...")
        print("Listening on default microphone. Ctrl+C to stop.\n")

        proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        proc_thread.start()

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
        )

        try:
            with stream:
                if duration:
                    time.sleep(duration)
                else:
                    while True:
                        time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            proc_thread.join(timeout=5)
            print("\nLive session stopped.")


# ===================================================================
# FASTAPI BACKEND (Phase 1-9 batch API + Phase 10 live status)
# ===================================================================

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import shutil

app = FastAPI(title="EdgeGD API", version="1.0")

_live_pipeline: Optional[StreamingPipeline] = None
_live_thread: Optional[threading.Thread] = None


@app.post("/analyze")
async def api_analyze(
    audio: UploadFile = File(...),
    topic: str = Form("Group Discussion"),
    num_speakers: Optional[int] = Form(None),
    save_to_db: bool = Form(False),
):
    """Phase 1-9: upload an audio file and receive full analytics + PDF."""
    upload_dir = os.path.join(Config.OUTPUT_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    audio_path = os.path.join(upload_dir, audio.filename)

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    try:
        result = analyze_session(
            audio_path=audio_path,
            topic=topic,
            num_speakers=num_speakers,
            save_to_db=save_to_db,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(_result_to_dict(result))


@app.get("/report/{session_id}")
async def api_get_report(session_id: str):
    """Download a previously generated PDF report."""
    path = os.path.join(Config.OUTPUT_DIR, f"report_{session_id}.pdf")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="application/pdf",
                         filename=f"edgegd_report_{session_id}.pdf")


@app.post("/live/start")
async def api_live_start(duration: Optional[float] = Form(None),
                          num_speakers: Optional[int] = Form(None)):
    """Phase 10: start a real-time session in the background."""
    global _live_pipeline, _live_thread

    if _live_thread and _live_thread.is_alive():
        raise HTTPException(status_code=400, detail="Live session already running")

    _live_pipeline = StreamingPipeline(num_speakers=num_speakers)
    _live_thread = threading.Thread(
        target=_live_pipeline.run, kwargs={"duration": duration}, daemon=True)
    _live_thread.start()
    return {"status": "started"}


@app.get("/live/status")
async def api_live_status():
    """Phase 10: poll current live metrics."""
    if _live_pipeline is None:
        raise HTTPException(status_code=404, detail="No live session running")
    return _live_pipeline.metrics_engine.current_metrics()


@app.post("/live/stop")
async def api_live_stop():
    global _live_pipeline, _live_thread
    if _live_pipeline is None:
        raise HTTPException(status_code=404, detail="No live session running")
    _live_pipeline._stop_event.set()
    _live_pipeline = None
    _live_thread = None
    return {"status": "stopped"}


# ===================================================================
# CLI ENTRY POINT
# ===================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="EdgeGD - Group Discussion Analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Run Phases 1-9 on a recorded audio file")
    p_analyze.add_argument("--audio", required=True, help="Path to audio file")
    p_analyze.add_argument("--topic", default="Group Discussion")
    p_analyze.add_argument("--speakers", type=int, default=None,
                            help="Expected number of speakers")
    p_analyze.add_argument("--save-db", action="store_true")
    p_analyze.add_argument("--output-dir", default=Config.OUTPUT_DIR)

    p_live = sub.add_parser("live", help="Run Phase 10 real-time analysis")
    p_live.add_argument("--duration", type=float, default=None,
                         help="Duration in seconds (omit for indefinite)")
    p_live.add_argument("--speakers", type=int, default=None)

    args = parser.parse_args()

    if args.command == "analyze":
        analyze_session(
            audio_path=args.audio,
            topic=args.topic,
            num_speakers=args.speakers,
            output_dir=args.output_dir,
            save_to_db=args.save_db,
        )
    elif args.command == "live":
        pipeline = StreamingPipeline(num_speakers=args.speakers)
        pipeline.run(duration=args.duration)


if __name__ == "__main__":
    main()