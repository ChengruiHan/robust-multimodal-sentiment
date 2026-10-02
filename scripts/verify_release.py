"""Check the frozen evidence manifest; no dataset or model dependencies."""
import argparse
from pathlib import Path

from release_evidence import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="External private evidence bundle directory")
    args = parser.parse_args()
    print(f"Verified {verify_inputs(args.root)} frozen evidence files (SHA-256).")


if __name__ == "__main__":
    main()
