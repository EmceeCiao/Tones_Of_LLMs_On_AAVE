#!/usr/bin/env python3
"""
build_dpo_from_qualtrics.py

Builds qualtrics_dpo_pairs.jsonl from the Round 1 Qualtrics human preference
study.  For each of the 15 prompts, it counts SAE vs AAVE votes and keeps only
those where the winning side leads by at least STRONG_MAJORITY_MIN_MARGIN votes.
"No Preference" responses count toward the total respondents but NOT toward
either side, so a 3-2 split with 4 No Preferences does NOT qualify as a strong
majority.

For each qualifying prompt two DPO records are emitted:
  - prompt_version="AAVE": prompt = AAVE form of the question
  - prompt_version="SAE":  prompt = SAE  form of the question

Usage:
    python mitigations/finetuning/build_dpo_from_qualtrics.py

No API calls. No extra dependencies beyond openpyxl (pip install openpyxl).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import openpyxl


# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

QUALTRICS_DPO_JSONL = SCRIPT_DIR / "qualtrics_dpo_pairs.jsonl"

FINAL_RESPONSES_JSON = REPO_ROOT / "dataset" / "Final_Dataset" / "final_prompt_responses.json"
QUALTRICS_XLSX = (
    REPO_ROOT
    / "Round_1_Results"
    / "LLM & AAVE ROUND 1_April 29, 2026_02.09_aggregated.xlsx"
)


# ── Threshold ────────────────────────────────────────────────────────────────
# A prompt qualifies only when the winning dialect leads by at least this many
# votes among opinionated respondents (SAE + AAVE only).  "No Preference" votes
# are excluded from both the numerator and denominator, but because they reduce
# the opinionated count a 3-2 split with 4 No Preferences still doesn't make
# the cut — the margin itself must be >= this value.
STRONG_MAJORITY_MIN_MARGIN = 3

# ── Helpers ──────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Step 2: Build qualtrics_dpo_pairs.jsonl ───────────────────────────────────

def load_qualtrics_votes() -> dict[str, dict[str, Any]]:
    """Return {prompt_id: {SAE: n, AAVE: n, NoPref: n, majority: str}} for P01–P15."""
    wb = openpyxl.load_workbook(QUALTRICS_XLSX)
    ws = wb["Aggregated"]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]

    results: dict[str, dict[str, Any]] = {}
    for i in range(1, 16):
        pid = f"P{i:02d}"
        col_name = f"{pid}_Preference"
        col_idx = headers.index(col_name) + 1
        votes = [ws.cell(r, col_idx).value for r in range(2, ws.max_row + 1)]

        counts = Counter(v for v in votes if v in ("SAE", "AAVE"))
        sae = counts.get("SAE", 0)
        aave = counts.get("AAVE", 0)
        no_pref = sum(1 for v in votes if v == "No Preference")

        if sae > aave:
            majority = "SAE"
        elif aave > sae:
            majority = "AAVE"
        else:
            majority = "TIE"

        results[pid] = {
            "sae_votes": sae,
            "aave_votes": aave,
            "no_pref_votes": no_pref,
            "majority": majority,
            "margin": abs(sae - aave),
        }
    return results


def load_round1_responses() -> dict[str, dict[str, Any]]:
    """Return {prompt_id: {category, sae_prompt, aave_prompt, sae_response, aave_response}}."""
    data = json.loads(FINAL_RESPONSES_JSON.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(data["round 1"], start=1):
        pid = f"P{idx:02d}"
        result[pid] = {
            "category": item.get("category", ""),
            "sae_prompt": item["sae_prompt"],
            "aave_prompt": item["aave_prompt"],
            "sae_response": item["sae_response"],
            "aave_response": item["aave_response"],
        }
    return result


def build_qualtrics_dpo_pairs() -> None:
    """Emit two DPO records per prompt that has a strong majority (AAVE-prompt + SAE-prompt)."""
    votes = load_qualtrics_votes()
    responses = load_round1_responses()

    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for pid in sorted(votes):
        v = votes[pid]
        if v["majority"] == "TIE" or v["margin"] < STRONG_MAJORITY_MIN_MARGIN:
            skipped.append(pid)
            continue

        r = responses[pid]
        majority = v["majority"]  # "SAE" or "AAVE"

        # chosen = the response humans preferred; rejected = the other
        if majority == "SAE":
            chosen_response = r["sae_response"]
            rejected_response = r["aave_response"]
        else:
            chosen_response = r["aave_response"]
            rejected_response = r["sae_response"]

        shared_meta = {
            "prompt_id": pid,
            "category": r["category"],
            "chosen_dialect": majority,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "sae_votes": v["sae_votes"],
            "aave_votes": v["aave_votes"],
            "no_pref_votes": v["no_pref_votes"],
            "margin": v["margin"],
            "source": "qualtrics_round1",
        }

        # AAVE-prompt version
        records.append({**shared_meta, "prompt_version": "AAVE", "prompt": r["aave_prompt"]})
        # SAE-prompt version
        records.append({**shared_meta, "prompt_version": "SAE", "prompt": r["sae_prompt"]})

    write_jsonl(QUALTRICS_DPO_JSONL, records)

    n_prompts = len(records) // 2
    print(f"qualtrics_dpo_pairs.jsonl: {len(records)} records from {n_prompts} prompts  ({QUALTRICS_DPO_JSONL})")
    if skipped:
        print(f"  Skipped (tie or weak majority, margin < {STRONG_MAJORITY_MIN_MARGIN}): {', '.join(skipped)}")

    # Print a quick breakdown
    print()
    print(f"  prompt  category    SAE  AAVE  NoPref  margin  result")
    print("  " + "-" * 58)
    for pid in sorted(votes):
        v = votes[pid]
        r = responses[pid]
        if v["majority"] == "TIE":
            tag = "SKIP(TIE)"
        elif v["margin"] < STRONG_MAJORITY_MIN_MARGIN:
            tag = f"SKIP(weak, margin={v['margin']})"
        else:
            tag = v["majority"]
        print(
            f"  {pid}    {r['category']:<12}"
            f"{v['sae_votes']:>3}  {v['aave_votes']:>4}  {v['no_pref_votes']:>6}"
            f"  {v['margin']:>6}  {tag}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Building qualtrics_dpo_pairs.jsonl ===")
    build_qualtrics_dpo_pairs()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
