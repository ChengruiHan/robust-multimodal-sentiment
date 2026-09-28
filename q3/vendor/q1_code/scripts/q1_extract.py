"""Adapters for frozen BERT, WhisperX, openSMILE and OpenFace."""
from __future__ import annotations

import csv
from functools import lru_cache
from importlib import metadata
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from .qtop import FrameSeries


AU_COLUMNS = [f"AU{x:02d}_r" for x in (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45)]
VISION_COLUMNS = AU_COLUMNS + ["pose_Rx", "pose_Ry", "pose_Rz"]


@lru_cache(maxsize=2)
def _align_model(language: str, device: str, model_name: str | None):
    import whisperx
    return whisperx.load_align_model(language_code=language, device=device, model_name=model_name)


@lru_cache(maxsize=2)
def _bert_model(model_name: str, revision: str, device: str):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
    return tokenizer, model


def decode_audio(video: Path, wav: Path, sr: int, ffmpeg: str) -> None:
    cmd = [ffmpeg, "-nostdin", "-y", "-v", "error", "-i", str(video), "-map", "0:a:0",
           "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"audio decode failed: {result.stderr.strip()}")


def audio_signal_stats(wav: Path) -> dict:
    """Measure the decoded PCM; stream presence alone does not imply speech."""
    import soundfile as sf
    samples, sample_rate = sf.read(wav, dtype="int16")
    if samples.ndim != 1:
        raise ValueError("Q1 audio signal audit requires mono PCM")
    values = samples.astype(np.float64) / 32768.0
    return {"audio_samples": int(len(samples)), "decoded_audio_sr": int(sample_rate),
            "audio_nonzero_count": int(np.count_nonzero(samples)),
            "audio_peak": float(np.max(np.abs(values))) if len(values) else 0.0,
            "audio_rms": float(np.sqrt(np.mean(values * values))) if len(values) else 0.0}


def align_words(words: list[dict], wav: Path, audio_start: float, audio_end: float,
                device: str, language: str, model_name: str | None,
                nltk_data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    if nltk_data_dir is not None:
        os.environ["NLTK_DATA"] = str(nltk_data_dir)
        import nltk
        nltk.data.path.insert(0, str(nltk_data_dir))
    import soundfile as sf
    import whisperx
    audio, sr = sf.read(wav, dtype="float32")
    if sr != 16000:
        raise ValueError("WhisperX alignment requires the configured 16 kHz wav")
    if audio.ndim != 1:
        raise ValueError("alignment audio must be mono")
    model, meta = _align_model(language, device, model_name)
    transcript = " ".join(w["align_word"] for w in words)
    result = whisperx.align([{"text": transcript, "start": 0.0, "end": len(audio) / sr}],
                            model, meta, audio, device, return_char_alignments=False)
    aligned = result.get("word_segments", [])
    starts = np.full(len(words), np.nan, dtype=np.float64)
    ends = np.full(len(words), np.nan, dtype=np.float64)
    statuses = ["unmatched"] * len(words)
    j = 0
    for i, word in enumerate(words):
        target = re.sub(r"[^a-z0-9]", "", word["align_word"])
        if not target:
            continue
        # Only match the next aligner token. A mismatch is explicit and does
        # not silently shift subsequent token-to-time associations.
        if j >= len(aligned):
            break
        candidate = re.sub(r"[^a-z0-9]", "", aligned[j].get("word", "").lower())
        if candidate != target:
            statuses[i] = "unmatched_token_mismatch"
            continue
        item = aligned[j]
        j += 1
        if "start" not in item or "end" not in item:
            statuses[i] = "unmatched_no_timestamp"
            continue
        s = float(item["start"]) + audio_start
        e = float(item["end"]) + audio_start
        if 0 <= s < e <= audio_end + 1e-4 and (i == 0 or not np.isfinite(ends[:i]).any() or s >= np.nanmax(ends[:i]) - 1e-3):
            starts[i], ends[i], statuses[i] = s, min(e, audio_end), "automatic"
        else:
            statuses[i] = "unmatched_invalid_time"
    return starts, ends, statuses, {"aligner": "whisperx", "align_model": model_name or "whisperx default",
                                   "aligner_version": metadata.version("whisperx"),
                                   "returned_words": len(aligned), "matched_words": sum(x == "automatic" for x in statuses)}


def bert_words(words: list[dict], model_name: str, revision: str, device: str) -> tuple[np.ndarray, list[list[int]], dict]:
    import torch
    tokenizer, model = _bert_model(model_name, revision, device)
    tokens = [w["align_word"] for w in words]
    encoded = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=False)
    if encoded["input_ids"].shape[1] > model.config.max_position_embeddings:
        raise ValueError("BERT input exceeds model context; no silent truncation permitted")
    with torch.inference_mode():
        h = model(**{k: v.to(device) for k, v in encoded.items()}).last_hidden_state[0].cpu().numpy()
    mapping = [[] for _ in words]
    for position, word_id in enumerate(encoded.word_ids()):
        if word_id is not None:
            mapping[word_id].append(position)
    if any(not ids for ids in mapping):
        raise ValueError("BERT tokenizer produced an empty subword mapping")
    features = np.stack([h[ids].mean(axis=0) for ids in mapping]).astype(np.float32)
    if features.shape[1] != 768 or not np.isfinite(features).all():
        raise ValueError(f"unexpected BERT features: {features.shape}")
    return features, mapping, {"model": model_name, "revision": revision,
                               "resolved_commit": getattr(model.config, "_commit_hash", None),
                               "transformers_version": metadata.version("transformers"),
                               "torch_version": metadata.version("torch")}


def audio_frames(wav: Path, stream_span: list[float]) -> tuple[FrameSeries, list[str], dict]:
    import opensmile
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                           feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    frame = smile.process_file(str(wav))
    columns = frame.columns.tolist()
    if len(columns) != 25:
        raise ValueError(f"eGeMAPSv02 LLD has {len(columns)} columns, expected 25")
    starts = frame.index.get_level_values("start").total_seconds().to_numpy()
    ends = frame.index.get_level_values("end").total_seconds().to_numpy()
    centers = stream_span[0] + (starts + ends) / 2
    values = frame.to_numpy(dtype=np.float32, copy=True)
    quality = np.isfinite(values).all(axis=1).astype(np.float64)
    values[quality == 0] = 0  # Invalid source frames are excluded by quality.
    series = FrameSeries(values, centers, quality, np.arange(len(values)), *stream_span)
    return series, columns, {"opensmile_version": metadata.version("opensmile"),
                             "feature_set": "eGeMAPSv02", "feature_level": "LowLevelDescriptors",
                             "frame_time_source": "openSMILE dataframe interval midpoint"}


def video_presentation_times(video: Path, ffprobe: str, container_start: float) -> np.ndarray:
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
           "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"video frame timestamp probe failed: {result.stderr.strip()}")
    frames = json.loads(result.stdout).get("frames", [])
    pts = np.asarray([float(f["best_effort_timestamp_time"]) - container_start for f in frames], dtype=np.float64)
    if not len(pts) or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("video frame PTS missing or nonmonotone")
    return pts


def video_frames(video: Path, work: Path, command: str, extra_args: list[str],
                 stream_span: list[float], confidence_mode: str, ffprobe: str,
                 container_start: float) -> tuple[FrameSeries, dict]:
    work.mkdir(parents=True, exist_ok=True)
    cmd = [command, "-f", str(video), "-out_dir", str(work), "-aus", "-pose", *extra_args]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(Path(command).parent) if Path(command).is_file() else None)
    if result.returncode:
        raise RuntimeError(f"OpenFace failed: {result.stderr[-1200:]}")
    csv_file = work / f"{video.stem}.csv"
    if not csv_file.exists():
        candidates = list(work.glob("*.csv"))
        if len(candidates) != 1:
            raise RuntimeError("OpenFace did not produce a unique CSV")
        csv_file = candidates[0]
    with csv_file.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = [{k.strip(): v for k, v in row.items()} for row in reader]
    if not rows:
        raise RuntimeError("OpenFace produced no frames")
    missing = set(VISION_COLUMNS + ["frame", "timestamp", "confidence", "success"]) - set(rows[0])
    if missing:
        raise ValueError(f"OpenFace missing columns: {sorted(missing)}")
    pts = video_presentation_times(video, ffprobe, container_start)
    frame_numbers = np.asarray([int(r["frame"]) for r in rows], dtype=np.int64)
    if np.any(frame_numbers < 1) or np.any(frame_numbers > len(pts)):
        raise ValueError("OpenFace frame number cannot be mapped to decoded video PTS")
    centers = pts[frame_numbers - 1]
    if np.any(np.diff(centers) <= 0):
        raise ValueError("OpenFace duplicate/nonmonotone timestamps; ambiguous face tracks")
    values = np.asarray([[float(r[c]) for c in VISION_COLUMNS] for r in rows], dtype=np.float32)
    confidence = np.asarray([float(r["confidence"]) for r in rows])
    success = np.asarray([float(r["success"]) == 1 for r in rows])
    if confidence_mode == "clipped":
        quality = success * np.clip(confidence, 0, 1)
    elif confidence_mode == "binary":
        quality = success.astype(float)
    else:
        raise ValueError("confidence_mode must be clipped or binary")
    quality[~np.isfinite(quality)] = 0
    quality[~np.isfinite(values).all(axis=1)] = 0
    values[~np.isfinite(values).all(axis=1)] = 0
    # Some MP4s declare a stream duration shorter than their last decoded PTS.
    # Preserve the measured frame instead of rejecting the entire visual stream.
    half_step = float(np.median(np.diff(centers)) / 2) if len(centers) > 1 else 0.02
    measured_span = (min(stream_span[0], float(centers[0])),
                     max(stream_span[1], float(centers[-1]) + half_step))
    series = FrameSeries(values, centers, quality, frame_numbers, *measured_span)
    return series, {"columns": VISION_COLUMNS, "openface_command": command,
                    "confidence_mode": confidence_mode,
                    "extra_args": extra_args,
                    "frame_time_source": "ffprobe best_effort_timestamp_time mapped by OpenFace frame number"}
