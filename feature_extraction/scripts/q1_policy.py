"""Small, auditable gate for risky transcript/audio pairs."""
from __future__ import annotations

import csv
from pathlib import Path


def load_review_files(candidates_path: Path, decisions_path: Path,
                      official_ids: set[str]) -> tuple[set[str], dict[str, dict]]:
    candidates: set[str] = set()
    with candidates_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sample_id = row["sample_id"].strip()
            if sample_id not in official_ids or sample_id in candidates:
                raise ValueError(f"invalid or duplicate review candidate: {sample_id}")
            candidates.add(sample_id)
    decisions: dict[str, dict] = {}
    with decisions_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sample_id = row["sample_id"].strip()
            decision = row["decision"].strip()
            if sample_id not in official_ids or sample_id in decisions:
                raise ValueError(f"invalid or duplicate review decision: {sample_id}")
            if decision not in {"ALLOW_ALIGN", "BLOCK_ALIGN"}:
                raise ValueError(f"invalid review decision for {sample_id}: {decision}")
            if not row["note"].strip():
                raise ValueError(f"review decision needs evidence note: {sample_id}")
            decisions[sample_id] = {"decision": decision, "note": row["note"].strip(),
                                    "evidence_time_s": row["evidence_time_s"].strip()}
    return candidates, decisions


def alignment_gate(audio_nonzero_count: int | None, sample_id: str,
                   candidates: set[str], decisions: dict[str, dict]) -> str:
    if audio_nonzero_count is None:
        return "SKIP_AUDIO_UNAVAILABLE"
    if audio_nonzero_count == 0:
        return "SKIP_SILENT"
    decision = decisions.get(sample_id, {}).get("decision")
    if decision == "BLOCK_ALIGN":
        return "BLOCKED"
    if sample_id in candidates and decision != "ALLOW_ALIGN":
        return "REVIEW_REQUIRED"
    return "AUTO_UNVERIFIED"
