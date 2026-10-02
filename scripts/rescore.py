"""Recompute metrics from an external private numerical evidence bundle."""
import argparse
import json
from pathlib import Path

from release_evidence import ROOT, rescore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="External private evidence bundle directory")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/local_rescore.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = rescore(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"state_msa": {k: result["state_msa"][k] for k in ("clean", "missing_mean")},
                      "trace_msa": result["trace_msa"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
