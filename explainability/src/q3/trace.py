"""Map aligned_50 token positions back to original text and video time.

Forced alignment is a frozen Q1 tool. Machine timestamps remain approximate
until reviewed; failures never receive fabricated times.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:[.,]\d+)*")


def words_with_spans(text: str) -> list[dict]:
    return [dict(word=m.group(), align_word=m.group().replace("’", "'").lower(),
                 char_start=m.start(), char_end=m.end()) for m in WORD_RE.finditer(text)]


def token_map(raw_text: str, bert: np.ndarray, tokenizer_path: str | Path) -> list[dict]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True, local_files_only=True)
    encoded = tokenizer(raw_text, truncation=True, max_length=50, return_offsets_mapping=True)
    expected = np.asarray(bert[0], dtype=np.int64)
    attended = int(np.asarray(bert[1]).sum())
    if encoded["input_ids"] != expected[:attended].tolist():
        raise ValueError("raw_text tokenization does not match attachment text_bert")
    words = words_with_spans(raw_text)
    rows = []
    for position, (start, end) in enumerate(encoded["offset_mapping"]):
        if position == 0 or position == attended - 1 or start == end:
            continue
        matches = [i for i, word in enumerate(words)
                   if max(start, word["char_start"]) < min(end, word["char_end"])]
        punctuation_only = False
        if not matches:
            # BERT keeps punctuation tokens, while the forced aligner only
            # timestamps spoken words. Associate punctuation with the nearest
            # preceding word (or the first following word at sentence start).
            before = [i for i, word in enumerate(words) if word["char_end"] <= start]
            after = [i for i, word in enumerate(words) if word["char_start"] >= end]
            matches = [before[-1] if before else after[0]] if (before or after) else []
            punctuation_only = True
        if len(matches) != 1:
            raise ValueError(f"token position {position} has ambiguous raw word mapping")
        rows.append(dict(position=position, char_start=int(start), char_end=int(end),
                         word_id=matches[0], token_text=raw_text[start:end],
                         punctuation_only=punctuation_only))
    return rows


def media_info(video: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                             "-of", "json", str(video)], capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)
    fmt = probe["format"]
    container_start = float(fmt.get("start_time", 0))
    duration = float(fmt["duration"])
    audio = next((x for x in probe.get("streams", []) if x.get("codec_type") == "audio"), None)
    if audio is None:
        raise ValueError("video has no audio stream")
    audio_start = max(0., float(audio.get("start_time", container_start)) - container_start)
    audio_end = min(duration, audio_start + float(audio.get("duration", duration - audio_start)))
    return dict(duration=duration, container_start=container_start,
                audio_start=audio_start, audio_end=audio_end)


def build_trace(raw_text: str, bert: np.ndarray, video: Path, tokenizer_path: str | Path,
                align: bool = False, device: str = "cpu", q1_root: str | Path | None = None) -> dict:
    """Return token rows and optional word timestamps/P. T. S. values."""
    tokens = token_map(raw_text, bert, tokenizer_path)
    words = words_with_spans(raw_text)
    info = media_info(video)
    starts = np.full(len(words), np.nan)
    ends = np.full(len(words), np.nan)
    statuses = ["alignment_not_run"] * len(words)
    pts = np.empty(0, dtype=np.float64)
    warning = "alignment_not_run"
    if align:
        root = Path(q1_root) if q1_root else Path(__file__).resolve().parents[3] / "feature_extraction"
        if not (root / "scripts" / "q1_extract.py").is_file():
            raise FileNotFoundError(f"Q1 code not found: {root}")
        sys.path.insert(0, str(root))
        from scripts.q1_extract import align_words, decode_audio, video_presentation_times
        nltk_data = Path(tokenizer_path).resolve().parent / "nltk_data"
        with tempfile.TemporaryDirectory(prefix="q3_align_") as directory:
            wav = Path(directory) / "audio.wav"
            decode_audio(video, wav, 16000, "ffmpeg")
            starts, ends, statuses, _ = align_words(words, wav, info["audio_start"],
                                                      info["audio_end"], device, "en", None,
                                                      nltk_data if nltk_data.is_dir() else None)
        try:
            pts = video_presentation_times(video, "ffprobe", info["container_start"])
        except (ValueError, RuntimeError) as exc:
            warning = f"video_pts_unavailable:{type(exc).__name__}"
        else:
            warning = "machine_alignment_unreviewed"
    return dict(raw_text=raw_text, tokens=tokens, words=words, starts=starts, ends=ends,
                statuses=statuses, pts=pts, duration=info["duration"], warning=warning)


def locate(trace: dict | None, evidence: dict) -> dict:
    """Add raw span and optional valid time/representative frame to evidence."""
    result = dict(text_char_start=None, text_char_end=None, text_span="",
                  start_sec=None, end_sec=None, keyframe_sec=None,
                  mapping_quality="unresolved", mapping_warning="trace_unavailable")
    if trace is None:
        return result
    covered = [t for t in trace["tokens"] if evidence["start_idx"] <= t["position"] < evidence["end_idx"]]
    if not covered:
        result["mapping_warning"] = "no_mapped_tokens"
        return result
    word_ids = sorted({t["word_id"] for t in covered})
    left = min(t["char_start"] for t in covered)
    right = max(t["char_end"] for t in covered)
    result.update(text_char_start=left, text_char_end=right,
                  text_span=trace.get("raw_text", " ".join(w["word"] for w in trace["words"]))[left:right])
    if evidence["modality"] == "text":
        result.update(mapping_quality="verified", mapping_warning="text_token_match")
    valid = all(trace["statuses"][i] == "automatic" and np.isfinite(trace["starts"][i])
                and np.isfinite(trace["ends"][i]) for i in word_ids)
    if not valid:
        if evidence["modality"] != "text":
            result["mapping_warning"] = "word_time_unavailable"
        return result
    start = float(np.min(trace["starts"][word_ids]))
    end = float(np.max(trace["ends"][word_ids]))
    if not (0 <= start < end <= trace["duration"] + 1e-3):
        result["mapping_warning"] = "invalid_time_bounds"
        return result
    result.update(start_sec=start, end_sec=end)
    if evidence["modality"] != "text":
        result.update(mapping_quality="approximate", mapping_warning="machine_alignment_unreviewed")
    if evidence["modality"] == "vision" and len(trace["pts"]):
        midpoint = (start + end) / 2
        candidates = trace["pts"][(trace["pts"] >= start) & (trace["pts"] < end)]
        if len(candidates):
            result["keyframe_sec"] = float(candidates[np.argmin(abs(candidates - midpoint))])
        else:
            result["mapping_warning"] = "no_video_frame_in_interval"
    return result
