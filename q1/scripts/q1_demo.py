"""Two measured Q1 figures: aligned speech and confirmed silence."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .q1_run import read_csv, resolve


def waveform(video: Path, ffmpeg: str) -> np.ndarray:
    result = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(video),
                             "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                             "-f", "f32le", "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype="<f4")


def frame_image(video: Path, ffmpeg: str, time_s: float, target: Path):
    from PIL import Image
    subprocess.run([ffmpeg, "-nostdin", "-y", "-v", "error", "-ss", str(time_s),
                    "-i", str(video), "-frames:v", "1", str(target)], check=True)
    return Image.open(target).convert("RGB")


def heatmap(ax, values: np.ndarray, xlim: tuple[float, float], label: str, note: str = "No valid mapped feature"):
    values = np.asarray(values, dtype=float)
    if values.size and np.isfinite(values).all() and np.any(np.abs(values) > 0):
        matrix = values.T
        matrix = np.clip((matrix - matrix.mean(axis=1, keepdims=True)) /
                         np.maximum(matrix.std(axis=1, keepdims=True), 1e-6), -2.5, 2.5)
        ax.imshow(matrix, aspect="auto", origin="lower", interpolation="nearest",
                  extent=[*xlim, 0, matrix.shape[0]], cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    else:
        ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center")
        ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.set_ylabel(label, fontsize=9)
    ax.tick_params(labelsize=8)


def render_case(cfg: dict, row: dict, kind: str, report: Path) -> Path:
    import matplotlib.pyplot as plt

    base = cfg["output_dir"] / "samples" / row["safe_id"]
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    meta = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    video = cfg["source_dir"] / row["video_relpath"]
    duration = float(row["duration_s"])
    wave = waveform(video, cfg["ffmpeg_command"])
    wave_times = np.arange(len(wave)) / 16000 + float(row["audio_start_s"] or 0)
    selected = (np.flatnonzero(data["alignment_valid"])[:8] if kind == "normal"
                else np.arange(min(8, len(meta["words"]))))
    if kind == "normal" and len(selected) < 3:
        raise ValueError("normal case requires at least three aligned words")
    if kind == "silent" and (row["alignment_status"] != "SKIP_SILENT" or data["alignment_valid"].any()):
        raise ValueError("silent case must contain no word-time mapping")

    fig = plt.figure(figsize=(16, 14), layout="constrained")
    grid = fig.add_gridspec(8, 1, height_ratios=[2, 1.2, 1.1, 1, 0.75, 1.1, 1, 1])
    frames = grid[0].subgridspec(1, 4, wspace=0.07)
    if kind == "normal":
        first = float(data["start_s"][selected[0]])
        last = float(data["end_s"][selected[-1]])
        frame_times = np.linspace(first, last, 4)
    else:
        frame_times = np.linspace(duration * 0.12, duration * 0.88, 4)
    for j, time_s in enumerate(frame_times):
        target = report / f"case_{kind}_frame_{j}.jpg"
        image = frame_image(video, cfg["ffmpeg_command"], float(time_s), target)
        ax = fig.add_subplot(frames[0, j])
        ax.imshow(image)
        ax.set_title(f"Video frame  t={time_s:.2f} s", fontsize=9)
        ax.axis("off")

    wave_ax, audio_ax, face_ax, words_ax, text_ax, aligned_audio_ax, aligned_face_ax = [
        fig.add_subplot(grid[i]) for i in range(1, 8)]
    stride = max(1, len(wave) // 9000)
    wave_ax.plot(wave_times[::stride], wave[::stride], lw=0.7, color="#455a64")
    wave_ax.set_xlim(0, duration)
    wave_ax.set_ylabel("PCM waveform", fontsize=9)
    wave_ax.grid(alpha=0.2)
    wave_ax.set_title("Measured source timeline; colored bands are official word intervals", loc="left", fontsize=10)
    for time_s in frame_times:
        wave_ax.axvline(time_s, color="#5c6bc0", ls=":", lw=0.8, alpha=0.75)
    if kind == "normal":
        for n, k in enumerate(selected):
            start, end = float(data["start_s"][k]), float(data["end_s"][k])
            wave_ax.axvspan(start, end, color="#bbdefb" if n % 2 == 0 else "#d1c4e9", alpha=0.7)
            wave_ax.text((start + end) / 2, 0.96, meta["words"][int(k)]["word"], rotation=45,
                         ha="center", va="top", transform=wave_ax.get_xaxis_transform(), fontsize=8)
    else:
        wave_ax.text(0.5, 0.55, "Exactly zero PCM: no trustworthy word time anchor",
                     transform=wave_ax.transAxes, ha="center", fontsize=11)

    for ax, prefix, label in ((audio_ax, "audio", "Audio eGeMAPS LLD 25D"),
                              (face_ax, "video", "OpenFace AU + pose 20D")):
        centers, values, quality = (data[f"{prefix}_raw_{suffix}"] for suffix in
                                    ("centers_s", "values", "quality"))
        valid = quality > 0
        if kind == "silent" and prefix == "audio":
            valid[:] = False
        if valid.any():
            heatmap(ax, values[valid], (float(centers[valid][0]), float(centers[valid][-1])), label)
        else:
            heatmap(ax, np.empty((0, values.shape[1])), (0, duration), label,
                    "No valid speech observation" if prefix == "audio" else "No valid face observation")
        ax.set_xlim(0, duration)
        ax.set_xlabel("Source video time (s)", fontsize=8)
        for time_s in frame_times:
            ax.axvline(time_s, color="#5c6bc0", ls=":", lw=0.6, alpha=0.5)

    words_ax.set_xlim(0, len(selected))
    words_ax.set_ylim(0, 1)
    words_ax.set_yticks([])
    words_ax.set_xticks([])
    words_ax.set_ylabel("Official words", fontsize=9)
    for n, k in enumerate(selected):
        color = "#eeeeee" if kind == "silent" else "#bbdefb" if n % 2 == 0 else "#d1c4e9"
        words_ax.axvspan(n, n + 1, color=color, alpha=0.8)
        words_ax.text(n + 0.5, 0.5, meta["words"][int(k)]["word"],
                      rotation=55, ha="center", va="center", fontsize=9)
    words_ax.set_title("No word-to-time mapping" if kind == "silent" else
                       "QTOP: overlap × frame quality → official word positions", loc="left", fontsize=10)

    heatmap(text_ax, data["text_features"][selected], (0, len(selected)), "BERT 768D")
    heatmap(aligned_audio_ax, data["audio_features"][selected], (0, len(selected)), "Audio QTOP 50D")
    heatmap(aligned_face_ax, data["visual_features"][selected], (0, len(selected)), "Face QTOP 40D")
    for ax in (text_ax, aligned_audio_ax, aligned_face_ax):
        ax.set_xticks(np.arange(len(selected)) + 0.5)
        ax.set_xticklabels([meta["words"][int(k)]["word"] for k in selected],
                           rotation=45, ha="right", fontsize=8)
    aligned_face_ax.set_xlabel("Official word index; display heatmaps normalized per feature", fontsize=9)
    fig.suptitle(f"Q1 {kind}: {row['video_id']}/{row['clip_id']} | {row['alignment_status']} | 768 / 50 / 40D",
                 fontsize=14)
    output = report / f"case_{kind}_timeline.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)

    details = []
    for k in selected:
        a0, a1 = data["audio_frame_offsets"][k:k + 2]
        v0, v1 = data["video_frame_offsets"][k:k + 2]
        details.append({"sample_id": row["sample_id"], "case": kind, "position": int(k),
                        "word": meta["words"][int(k)]["word"],
                        "char_start": meta["words"][int(k)]["char_start"],
                        "char_end": meta["words"][int(k)]["char_end"],
                        "start_s": data["start_s"][k], "end_s": data["end_s"][k],
                        "alignment_valid": bool(data["alignment_valid"][k]),
                        "audio_frame_ids": data["audio_frame_ids"][a0:a1].tolist(),
                        "video_frame_ids": data["video_frame_ids"][v0:v1].tolist(),
                        "audio_available": bool(data["audio_available"][k]),
                        "visual_available": bool(data["visual_available"][k]),
                        "audio_coverage": float(data["audio_coverage"][k]),
                        "visual_coverage": float(data["visual_coverage"][k])})
    with (report / f"case_{kind}_positions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    return output


def main() -> None:
    cache = Path(tempfile.gettempdir()) / "e-q1-matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/q1.yaml")
    parser.add_argument("--normal-id", default="-3g5yACwYnA$_$13")
    parser.add_argument("--silent-id", default="-mJ2ud6oKI8$_$1")
    args = parser.parse_args()
    cfg = resolve(Path(args.config).resolve())
    rows = {row["sample_id"]: row for row in read_csv(cfg["output_dir"] / "manifest.csv")}
    report = cfg["output_dir"] / "reports"
    report.mkdir(exist_ok=True)
    for kind, sample_id in (("normal", args.normal_id), ("silent", args.silent_id)):
        row = rows[sample_id]
        if row["status"] not in ("OK", "WARN"):
            raise ValueError(f"case {sample_id} has not been extracted")
        print(render_case(cfg, row, kind, report))


if __name__ == "__main__":
    main()
