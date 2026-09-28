"""Configure a copied Q1 package without editing the source project config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a portable Q1 config for the current machine")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="directory containing label-100.xlsx and video_id/clip_id.mp4")
    parser.add_argument("--openface", required=True, type=Path,
                        help="OpenFace 2.2.0 FeatureExtraction executable; model/ and AU_predictors/ must be beside it")
    parser.add_argument("--output-dir", type=Path, default=Path("q1_output"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source_dir.expanduser().resolve()
    executable = args.openface.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_dir() or not (source / "label-100.xlsx").is_file():
        parser.error("--source-dir must contain label-100.xlsx and the official video folders")
    if not executable.is_file() or not (executable.parent / "model/main_clnf_general.txt").is_file():
        parser.error("--openface needs FeatureExtraction and sibling model/main_clnf_general.txt")
    if not (executable.parent / "AU_predictors").is_dir():
        parser.error("OpenFace AU_predictors/ directory is missing beside FeatureExtraction")
    config = json.loads((root / "configs/q1.template.yaml").read_text(encoding="utf-8"))
    config.update(source_dir=str(source), labels=str(source / "label-100.xlsx"),
                  output_dir=str(output), openface_command=str(executable), device=args.device)
    destination = root / "configs/q1.yaml"
    destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
