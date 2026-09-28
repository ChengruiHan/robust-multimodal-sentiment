"""Single entry point for Q1 audit, extraction and validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

from .q1_sources import (decode_ok, ffprobe, load_official_labels, media_summary,
                         official_words, sha256)
from .q1_extract import (VISION_COLUMNS, align_words, audio_frames, audio_signal_stats, bert_words,
                         decode_audio, video_frames)
from .q1_policy import alignment_gate, load_review_files
from .qtop import aggregate


FIELDS = ["sample_id", "video_id", "clip_id", "video_relpath", "safe_id", "source_sha256",
          "duration_s", "video_start_s", "video_end_s", "audio_start_s", "audio_end_s",
          "video_valid_duration_s", "audio_valid_duration_s",
          "fps", "audio_sr", "video_decode_ok", "audio_decode_ok", "transcript_present",
          "source_status", "source_error", "words", "aligned_words", "text_dim", "audio_dim",
          "vision_dim", "text_status", "audio_status", "vision_status", "alignment_granularity",
          "audio_coverage", "vision_coverage", "align_rate", "audio_peak", "audio_rms",
          "audio_nonzero_count", "audio_signal_status", "alignment_status", "review_flag",
          "review_decision", "feature_version", "run_fingerprint", "status", "error_code"]

SCHEMA = {"feature_version": "qtop-v2-final", "alignment_unit": "official transcript word",
          "interval": "half-open seconds", "sequence": "variable length, no padding/truncation",
          "text_dim": 768, "audio_raw_dim": 25, "audio_dim": 50,
          "visual_raw_dim": 20, "visual_dim": 40, "visual_kind": "OpenFace face behavior",
          "vision_columns": VISION_COLUMNS, "frame_mapping": "CSR offsets and source IDs in NPZ",
          "time_source": "audio: openSMILE interval; video: ffprobe frame PTS"}


def write_schema(out: Path) -> None:
    (out / "feature_schema.json").write_text(json.dumps(SCHEMA, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for kind in ("video", "audio"):
            start, end = row.get(f"{kind}_start_s"), row.get(f"{kind}_end_s")
            row[f"{kind}_valid_duration_s"] = max(0.0, float(end) - float(start)) if start not in (None, "") and end not in (None, "") else 0.0
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def resolve(config_path: Path) -> dict:
    # JSON is a YAML 1.2 subset and needs no parser dependency for auditing.
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("source_dir", "labels", "output_dir", "review_candidates", "review_decisions"):
        cfg[key] = (config_path.parent / cfg[key]).resolve()
    for key in ("bert_model", "nltk_data_dir"):
        if key in cfg and isinstance(cfg[key], str) and cfg[key].startswith("."):
            cfg[key] = str((config_path.parent / cfg[key]).resolve())
    return cfg


def run_fingerprint(cfg: dict) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(SCHEMA, sort_keys=True).encode())
    for name in ("bert_revision", "audio_sample_rate", "whisperx_language", "whisperx_align_model",
                 "openface_command", "openface_extra_args", "openface_confidence_mode"):
        digest.update(json.dumps(cfg.get(name), sort_keys=True).encode())
    for name in ("review_candidates", "review_decisions"):
        digest.update(Path(cfg[name]).read_bytes())
    for name in ("q1_sources.py", "q1_extract.py", "q1_policy.py", "qtop.py", "q1_run.py"):
        digest.update(sha256(Path(__file__).parent / name).encode())
    return digest.hexdigest()


def run_version(command: list[str]) -> str:
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return "unavailable"


def write_tool_versions(cfg: dict) -> None:
    versions = {"python": platform.python_version(),
                "ffmpeg": run_version([cfg["ffmpeg_command"], "-version"]),
                "ffprobe": run_version([cfg["ffprobe_command"], "-version"]),
                "openface": run_version([cfg["openface_command"], "-help"]),
                "openface_binary_sha256": sha256(Path(cfg["openface_command"])),
                "bert_revision": cfg["bert_revision"],
                "alignment_model": cfg["whisperx_align_model"] or "whisperx English default",
                "feature_version": SCHEMA["feature_version"],
                "run_fingerprint": run_fingerprint(cfg),
                "project_code_sha256": {name: sha256(Path(__file__).parent / name) for name in
                                        ("q1_sources.py", "q1_extract.py", "q1_policy.py", "qtop.py", "q1_run.py")}}
    for package in ("numpy", "torch", "transformers", "whisperx", "opensmile", "soundfile", "openpyxl"):
        versions[package] = metadata.version(package)
    model_files = {"bert_weight_sha256": Path(cfg["bert_model"]) / "model.safetensors",
                   "whisperx_align_weight_sha256": Path.home() / ".cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth",
                   "openface_clnf_model_sha256": Path(cfg["openface_command"]).parent / "model/main_clnf_general.txt"}
    for name, path in model_files.items():
        versions[name] = sha256(path) if path.is_file() else "unavailable"
    (cfg["output_dir"] / "tool_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


def preflight(cfg: dict) -> list[str]:
    missing = [name for name in ("torch", "transformers", "opensmile", "whisperx", "soundfile")
               if importlib.util.find_spec(name) is None]
    errors = [f"Python package unavailable: {name}" for name in missing]
    for key in ("ffmpeg_command", "ffprobe_command", "openface_command"):
        if shutil.which(cfg[key]) is None and not Path(cfg[key]).is_file():
            errors.append(f"executable unavailable: {cfg[key]}")
    if cfg["audio_sample_rate"] != 16000:
        errors.append("WhisperX adapter requires audio_sample_rate=16000")
    if not (Path(cfg["bert_model"]) / "model.safetensors").is_file():
        errors.append("local BERT weights unavailable")
    if not (Path(cfg["nltk_data_dir"]) / "tokenizers" / "punkt_tab" / "english").is_dir():
        errors.append("NLTK punkt_tab English data unavailable")
    for key in ("review_candidates", "review_decisions"):
        if not Path(cfg[key]).is_file():
            errors.append(f"review file unavailable: {cfg[key]}")
    return errors


def audit(cfg: dict, decode: bool = True) -> list[dict]:
    labels = load_official_labels(cfg["labels"])
    if len(labels) != cfg["expected_samples"]:
        raise ValueError(f"expected {cfg['expected_samples']} labels, got {len(labels)}")
    expected = {f"{r['video_id']}/{r['clip_id']}.mp4" for r in labels}
    actual = {p.relative_to(cfg["source_dir"]).as_posix() for p in cfg["source_dir"].glob("*/*.mp4")}
    if expected != actual:
        raise ValueError(f"video key mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    rows = []
    for index, label in enumerate(labels, 1):
        rel = f"{label['video_id']}/{label['clip_id']}.mp4"
        path = cfg["source_dir"] / rel
        row = {**label, "video_relpath": rel, "source_sha256": sha256(path),
               "transcript_present": bool(label["raw_text"].strip()), "status": "AUDITED",
               "source_status": "OK", "source_error": "", "error_code": "",
               "words": len(official_words(label["raw_text"])), "aligned_words": 0,
               "text_dim": 0, "audio_dim": 0, "vision_dim": 0,
               "text_status": "PENDING", "audio_status": "PENDING", "vision_status": "PENDING",
               "alignment_granularity": "official_word",
               "audio_coverage": 0, "vision_coverage": 0, "align_rate": 0,
               "audio_peak": "", "audio_rms": "", "audio_nonzero_count": "",
               "audio_signal_status": "PENDING", "alignment_status": "PENDING",
               "review_flag": False, "review_decision": "", "feature_version": SCHEMA["feature_version"],
               "run_fingerprint": ""}
        try:
            info = media_summary(ffprobe(path, cfg["ffprobe_command"]))
            row.update({"duration_s": info["duration_s"], "fps": info["fps"], "audio_sr": info["audio_sr"],
                        "video_start_s": (info["video_span_s"] or [""])[0],
                        "video_end_s": (info["video_span_s"] or ["", ""])[1],
                        "audio_start_s": (info["audio_span_s"] or [""])[0],
                        "audio_end_s": (info["audio_span_s"] or ["", ""])[1]})
            row["video_decode_ok"] = bool(info["video_span_s"]) and (decode_ok(path, "video", cfg["ffmpeg_command"]) if decode else True)
            row["audio_decode_ok"] = bool(info["audio_span_s"]) and (decode_ok(path, "audio", cfg["ffmpeg_command"]) if decode else True)
            if not row["video_decode_ok"] or not row["audio_decode_ok"] or not row["transcript_present"]:
                row["source_status"] = "WARN"
                row["source_error"] = "MISSING_OR_UNDECODABLE_MODALITY_OR_TEXT"
            if row["audio_decode_ok"]:
                with tempfile.TemporaryDirectory(prefix="q1_audit_") as tmp:
                    wav = Path(tmp) / "audio.wav"
                    decode_audio(path, wav, cfg["audio_sample_rate"], cfg["ffmpeg_command"])
                    stats = audio_signal_stats(wav)
                row.update({k: stats[k] for k in ("audio_peak", "audio_rms", "audio_nonzero_count")})
                row["audio_signal_status"] = "SILENT" if stats["audio_nonzero_count"] == 0 else "NONZERO"
            else:
                row["audio_signal_status"] = "UNAVAILABLE"
        except Exception as exc:
            row["source_status"] = "FAIL"
            row["source_error"] = f"PROBE_FAILED:{type(exc).__name__}:{exc}"
            row["video_decode_ok"] = False
            row["audio_decode_ok"] = False
        rows.append(row)
        logging.info("audit %s/%s %s", index, len(labels), label["sample_id"])
    out = cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "manifest_source.csv", rows)
    write_csv(out / "manifest.csv", rows)
    write_schema(out)
    (out / "run_config.yaml").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
    write_tool_versions(cfg)
    return rows


def empty_qtop(length: int, dim: int) -> dict:
    return {"features": np.zeros((length, 2 * dim), dtype=np.float32),
            "available": np.zeros(length, bool), "coverage": np.zeros(length, np.float32),
            "quality": np.zeros(length, np.float32), "dispersion_valid": np.zeros(length, bool),
            "frame_offsets": np.zeros(length + 1, np.int64), "frame_ids": np.empty(0, np.int64),
            "support_cells": np.empty((0, 2), np.float64)}


def process_sample(cfg: dict, row: dict, candidates: set[str], decisions: dict[str, dict]) -> dict:
    source = cfg["source_dir"] / row["video_relpath"]
    if sha256(source) != row["source_sha256"]:
        raise ValueError("source video hash changed since audit")
    if row["source_status"] == "FAIL":
        raise ValueError(row["source_error"])
    info = media_summary(ffprobe(source, cfg["ffprobe_command"]))
    raw_text = next(x["raw_text"] for x in load_official_labels(cfg["labels"]) if x["sample_id"] == row["sample_id"])
    words = official_words(raw_text)
    if not words:
        raise ValueError("official transcript has no words")
    warnings = []
    starts = np.full(len(words), np.nan, np.float64)
    ends = np.full(len(words), np.nan, np.float64)
    statuses = ["unmatched"] * len(words)
    text, subword_map, text_meta = bert_words(words, cfg["bert_model"], cfg["bert_revision"], cfg["device"])
    audio_result = empty_qtop(len(words), 25)
    vision_result = empty_qtop(len(words), 20)
    audio_columns = []
    audio_meta = {}
    vision_meta = {}
    audio_series = None
    vision_series = None
    audio_extract_ok = False
    visual_extract_ok = False
    align_meta = {"status": "not_attempted"}
    stats = None
    alignment_status = "SKIP_AUDIO_UNAVAILABLE"
    with tempfile.TemporaryDirectory(prefix="q1_") as tmp:
        work = Path(tmp)
        if info["audio_span_s"]:
            try:
                wav = work / "audio.wav"
                decode_audio(source, wav, cfg["audio_sample_rate"], cfg["ffmpeg_command"])
                stats = audio_signal_stats(wav)
                alignment_status = alignment_gate(stats["audio_nonzero_count"], row["sample_id"], candidates, decisions)
                if alignment_status == "AUTO_UNVERIFIED":
                    try:
                        starts, ends, statuses, align_meta = align_words(words, wav, *info["audio_span_s"],
                            cfg["device"], cfg["whisperx_language"], cfg["whisperx_align_model"],
                            Path(cfg["nltk_data_dir"]))
                        if not np.isfinite(starts).any():
                            alignment_status = "ALIGN_FAILED"
                    except Exception as exc:
                        alignment_status = "ALIGN_FAILED"
                        align_meta = {"error": f"ALIGNMENT_FAILED:{type(exc).__name__}:{exc}"}
                        warnings.append(align_meta["error"])
                else:
                    align_meta = {"status": alignment_status, "reason": "signal_or_review_gate"}
                try:
                    audio_series, audio_columns, audio_meta = audio_frames(wav, info["audio_span_s"])
                    audio_result = aggregate(starts, ends, np.isfinite(starts), audio_series, 25)
                    audio_extract_ok = True
                except Exception as exc:
                    warnings.append(f"AUDIO_FEATURE_FAILED:{type(exc).__name__}:{exc}")
            except Exception as exc:
                alignment_status = "SKIP_AUDIO_UNAVAILABLE"
                align_meta = {"error": f"AUDIO_DECODE_FAILED:{type(exc).__name__}:{exc}"}
                warnings.append(align_meta["error"])
        else:
            align_meta = {"error": "NO_AUDIO_STREAM"}
            warnings.append("NO_AUDIO_STREAM")
        if info["video_span_s"]:
            try:
                vision_series, vision_meta = video_frames(source, work / "openface", cfg["openface_command"],
                    cfg["openface_extra_args"], info["video_span_s"], cfg["openface_confidence_mode"],
                    cfg["ffprobe_command"], info["container_start_s"])
                vision_result = aggregate(starts, ends, np.isfinite(starts), vision_series, 20)
                visual_extract_ok = True
            except Exception as exc:
                warnings.append(f"VISION_FEATURE_FAILED:{type(exc).__name__}:{exc}")
        else:
            warnings.append("NO_VIDEO_STREAM")
    valid = np.isfinite(starts) & np.isfinite(ends)
    if alignment_status not in ("AUTO_UNVERIFIED", "ALIGN_FAILED") and valid.any():
        raise AssertionError("review or signal gate produced word timestamps")
    out = cfg["output_dir"] / "samples"
    out.mkdir(parents=True, exist_ok=True)
    base = out / row["safe_id"]
    temp_npz = out / f".{row['safe_id']}.tmp.npz"
    temp_json = out / f".{row['safe_id']}.tmp.json"
    np.savez_compressed(temp_npz,
        text_features=text, audio_features=audio_result["features"], visual_features=vision_result["features"],
        start_s=starts, end_s=ends, alignment_valid=valid, sequence_valid=np.ones(len(words), bool),
        text_available=np.ones(len(words), bool),
        audio_available=audio_result["available"], visual_available=vision_result["available"],
        audio_coverage=audio_result["coverage"], visual_coverage=vision_result["coverage"],
        audio_quality=audio_result["quality"], visual_quality=vision_result["quality"],
        audio_dispersion_valid=audio_result["dispersion_valid"],
        visual_dispersion_valid=vision_result["dispersion_valid"],
        audio_frame_offsets=audio_result["frame_offsets"], audio_frame_ids=audio_result["frame_ids"],
        video_frame_offsets=vision_result["frame_offsets"], video_frame_ids=vision_result["frame_ids"],
        audio_raw_values=audio_series.values if audio_series is not None else np.empty((0, 25), np.float32),
        audio_raw_centers_s=audio_series.centers if audio_series is not None else np.empty(0, np.float64),
        audio_raw_quality=audio_series.quality if audio_series is not None else np.empty(0, np.float64),
        audio_raw_ids=audio_series.source_ids if audio_series is not None else np.empty(0, np.int64),
        audio_support_cells_s=audio_result["support_cells"],
        video_raw_values=vision_series.values if vision_series is not None else np.empty((0, 20), np.float32),
        video_raw_centers_s=vision_series.centers if vision_series is not None else np.empty(0, np.float64),
        video_raw_quality=vision_series.quality if vision_series is not None else np.empty(0, np.float64),
        video_raw_ids=vision_series.source_ids if vision_series is not None else np.empty(0, np.int64),
        video_support_cells_s=vision_result["support_cells"])
    mapping = {"sample_id": row["sample_id"], "video_id": row["video_id"], "clip_id": row["clip_id"],
               "video_relpath": row["video_relpath"], "source_file_hash": row["source_sha256"],
               "raw_text": raw_text, "align_text": " ".join(w["align_word"] for w in words),
               "words": words, "subword_mapping": subword_map, "alignment_status": statuses,
               "audio_source_frame_ids": [audio_result["frame_ids"][audio_result["frame_offsets"][i]:audio_result["frame_offsets"][i+1]].tolist() for i in range(len(words))],
               "video_source_frame_ids": [vision_result["frame_ids"][vision_result["frame_offsets"][i]:vision_result["frame_offsets"][i+1]].tolist() for i in range(len(words))],
               "audio_columns": audio_columns, "vision_columns": VISION_COLUMNS,
               "visual_feature_type": "OpenFace face behavior", "feature_version": SCHEMA["feature_version"],
               "run_fingerprint": run_fingerprint(cfg), "alignment_sample_status": alignment_status,
               "review_flag": row["sample_id"] in candidates,
               "review_decision": decisions.get(row["sample_id"]),
               "audio_signal": stats,
               "media": info, "text_model": text_meta, "alignment_model": align_meta,
               "audio_tool": audio_meta, "vision_tool": vision_meta, "warnings": warnings}
    temp_json.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_npz, base.with_suffix(".npz"))
    os.replace(temp_json, base.with_suffix(".json"))
    visual_has_face = visual_extract_ok and vision_series is not None and bool(np.any(vision_series.quality > 0))
    row.update({"aligned_words": int(valid.sum()), "text_dim": 768, "audio_dim": 50 if audio_extract_ok else 0,
                "vision_dim": 40 if visual_extract_ok else 0,
                "text_status": "OK", "audio_status": ("SILENT" if stats is not None and stats["audio_nonzero_count"] == 0
                                                       else "OK" if audio_extract_ok else "FAIL"),
                "vision_status": "OK" if visual_has_face else "NO_FACE" if visual_extract_ok else "FAIL",
                "audio_coverage": float(audio_result["coverage"].mean()),
                "vision_coverage": float(vision_result["coverage"].mean()),
                "align_rate": float(valid.mean()),
                "audio_peak": stats["audio_peak"] if stats else "",
                "audio_rms": stats["audio_rms"] if stats else "",
                "audio_nonzero_count": stats["audio_nonzero_count"] if stats else "",
                "audio_signal_status": "SILENT" if stats is not None and stats["audio_nonzero_count"] == 0
                                       else "NONZERO" if stats else "UNAVAILABLE",
                "alignment_status": alignment_status,
                "review_flag": row["sample_id"] in candidates,
                "review_decision": decisions.get(row["sample_id"], {}).get("decision", ""),
                "feature_version": SCHEMA["feature_version"], "run_fingerprint": run_fingerprint(cfg),
                "status": "OK" if not warnings and alignment_status == "AUTO_UNVERIFIED" and visual_has_face else "WARN",
                "error_code": ";".join(warnings)})
    return row


def validate(cfg: dict, sample_id: str | None = None) -> list[str]:
    out = cfg["output_dir"]
    rows = read_csv(out / "manifest.csv")
    errors = []
    if len(rows) != cfg["expected_samples"] or len({r["sample_id"] for r in rows}) != len(rows):
        errors.append("manifest count or sample IDs invalid")
    for row in rows:
        if sample_id and row["sample_id"] != sample_id:
            continue
        source = cfg["source_dir"] / row["video_relpath"]
        if sha256(source) != row["source_sha256"]:
            errors.append(f"{row['sample_id']}: source hash mismatch")
        if row["status"] == "AUDITED":
            errors.append(f"{row['sample_id']}: extraction pending")
            continue
        if row["status"] == "FAIL":
            errors.append(f"{row['sample_id']}: extraction failed: {row['error_code']}")
            continue
        if row["status"] not in ("OK", "WARN"):
            errors.append(f"{row['sample_id']}: unknown status {row['status']}")
            continue
        base = out / "samples" / row["safe_id"]
        try:
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as x:
                l = len(x["start_s"])
                if x["text_features"].shape != (l, 768) or x["audio_features"].shape != (l, 50) or x["visual_features"].shape != (l, 40):
                    raise ValueError("feature shape mismatch")
                if not x["sequence_valid"].all():
                    raise ValueError("canonical sequence contains padding")
                for name in ("text_features", "audio_features", "visual_features"):
                    if not np.isfinite(x[name]).all():
                        raise ValueError(f"nonfinite {name}")
                v = x["alignment_valid"]
                s, e = x["start_s"], x["end_s"]
                if np.any(v & (~np.isfinite(s) | ~np.isfinite(e) | (s < 0) | (s >= e) | (e > float(row["duration_s"]) + 1e-3))):
                    raise ValueError("invalid word times")
                if np.any(np.diff(s[v]) < -1e-3):
                    raise ValueError("nonmonotone word times")
                if np.any(~v & (np.isfinite(s) | np.isfinite(e))):
                    raise ValueError("invalid word has a finite timestamp")
                gate = row["alignment_status"]
                if gate in ("SKIP_SILENT", "SKIP_AUDIO_UNAVAILABLE", "REVIEW_REQUIRED", "BLOCKED", "ALIGN_FAILED") and v.any():
                    raise ValueError("gated or failed sample contains aligned words")
                if gate == "SKIP_SILENT" and (row["audio_signal_status"] != "SILENT" or int(row["audio_nonzero_count"]) != 0):
                    raise ValueError("silent gate disagrees with signal statistics")
                for prefix in ("audio", "visual"):
                    c = x[f"{prefix}_coverage"]
                    if np.any((c < 0) | (c > 1 + 1e-5)):
                        raise ValueError(f"invalid {prefix} coverage")
                    if np.any(~x[f"{prefix}_available"] & np.any(x[f"{prefix}_features"] != 0, axis=1)):
                        raise ValueError(f"unavailable {prefix} has nonzero feature")
                    key = "audio" if prefix == "audio" else "video"
                    offsets, ids = x[f"{key}_frame_offsets"], x[f"{key}_frame_ids"]
                    if len(offsets) != l + 1 or offsets[0] != 0 or offsets[-1] != len(ids) or np.any(np.diff(offsets) < 0):
                        raise ValueError(f"invalid {key} CSR mapping")
                    if np.any(x[f"{prefix}_available"] != (np.diff(offsets) > 0)):
                        raise ValueError(f"{key} availability and mapping disagree")
                    raw = x[f"{key}_raw_values"]
                    centers = x[f"{key}_raw_centers_s"]
                    quality = x[f"{key}_raw_quality"]
                    raw_ids = x[f"{key}_raw_ids"]
                    cells = x[f"{key}_support_cells_s"]
                    expected_dim = 25 if key == "audio" else 20
                    if raw.shape != (len(centers), expected_dim) or any(len(a) != len(centers) for a in (quality, raw_ids, cells)):
                        raise ValueError(f"invalid {key} raw frame shapes")
                    if not np.isfinite(raw).all() or not np.isfinite(centers).all() or not np.isfinite(quality).all():
                        raise ValueError(f"nonfinite {key} raw frame")
                    if np.any(np.diff(centers) <= 0) or np.any((quality < 0) | (quality > 1)):
                        raise ValueError(f"invalid {key} frame time/quality")
                    if not set(ids.tolist()).issubset(set(raw_ids.tolist())):
                        raise ValueError(f"{key} CSR references absent frame")
            with base.with_suffix(".json").open(encoding="utf-8") as f:
                meta = json.load(f)
            if meta["sample_id"] != row["sample_id"] or len(meta["words"]) != l:
                raise ValueError("JSON mapping mismatch")
            if meta["feature_version"] != SCHEMA["feature_version"] or meta["run_fingerprint"] != run_fingerprint(cfg):
                raise ValueError("stale feature or review decision")
        except Exception as exc:
            errors.append(f"{row['sample_id']}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/q1.yaml")
    parser.add_argument("--stage", choices=["audit", "preflight", "run", "validate"], default="run")
    parser.add_argument("--sample-id", help="retry only this audited sample")
    parser.add_argument("--no-decode", action="store_true", help="probe only, for fast source inspection")
    args = parser.parse_args()
    cfg = resolve(Path(args.config).resolve())
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(cfg["output_dir"] / "processing.log"), logging.StreamHandler()])
    if args.stage == "audit":
        audit(cfg, decode=not args.no_decode)
        return 0
    if args.stage in ("preflight", "run"):
        issues = preflight(cfg)
        for issue in issues:
            logging.error(issue)
        if issues:
            return 2
        if args.stage == "preflight":
            logging.info("Q1 extractor preflight passed")
            return 0
    if args.stage == "validate":
        errors = validate(cfg, args.sample_id)
        for error in errors:
            logging.error(error)
        logging.info("validation: %s errors", len(errors))
        return int(bool(errors))
    rows = read_csv(cfg["output_dir"] / "manifest.csv") if (cfg["output_dir"] / "manifest.csv").exists() else audit(cfg)
    candidates, decisions = load_review_files(cfg["review_candidates"], cfg["review_decisions"],
                                               {r["sample_id"] for r in rows})
    fingerprint = run_fingerprint(cfg)
    if args.sample_id and args.sample_id not in {r["sample_id"] for r in rows}:
        parser.error("sample-id not in official manifest")
    for row in rows:
        if args.sample_id and row["sample_id"] != args.sample_id:
            continue
        base = cfg["output_dir"] / "samples" / row["safe_id"]
        if (base.with_suffix(".npz").exists() and base.with_suffix(".json").exists()
                and row.get("run_fingerprint") == fingerprint and row.get("feature_version") == SCHEMA["feature_version"]
                and row["status"] in ("OK", "WARN") and not cfg["overwrite"] and not args.sample_id):
            continue
        try:
            process_sample(cfg, row, candidates, decisions)
            logging.info("processed %s: %s / %s", row["sample_id"], row["status"], row["alignment_status"])
        except Exception as exc:
            row["status"] = "FAIL"
            row["error_code"] = f"{type(exc).__name__}:{exc}"
            row["text_status"] = "FAIL"
            row["audio_status"] = "FAIL"
            row["vision_status"] = "FAIL"
            logging.error("%s: %s\n%s", row["sample_id"], exc, traceback.format_exc())
        write_csv(cfg["output_dir"] / "manifest.csv", rows)
        write_csv(cfg["output_dir"] / "q1_all_100.csv", rows)
    write_schema(cfg["output_dir"])
    errors = validate(cfg, args.sample_id)
    for error in errors:
        logging.error(error)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
