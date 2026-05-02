#!/usr/bin/env python3
"""
build_qualtrics_json.py

Takes any study-log JSONL (Round 1 or Round 2 format) and produces a
Qualtrics-ready JSON file matching the Round_1_Qualtrics.json schema.

Output JSON schema (one object per prompt):
  {
    "category":   "ELI5",
    "prompt":     "<SAE prompt text>",
    "response_A": "<SAE response text>",
    "response_B": "<AAVE response text>",
    "label_A":    "SAE",
    "label_B":    "AAVE"
  }

Then run the HTML converter separately:
    python scripts/qualtrics_html_converter_v5.py \\
        Final_Dataset/Qualtrics/Round_2_Qualtrics.json \\
        --fields prompt response_A response_B \\
        --output-json  Final_Dataset/Qualtrics/Round_2_Qualtrics_html.json \\
        --preview-html Final_Dataset/Qualtrics/Round_2_preview.html \\
        --snippet-dir  Final_Dataset/Qualtrics/Round_2_snippets

Usage:
    # Round 2:
    python scripts/build_qualtrics_json.py \\
        study_logs_round2/round2_20260502T010128Z.jsonl \\
        --output Final_Dataset/Qualtrics/Round_2_Qualtrics.json

    # Round 1 (filter by round when the log has mixed rounds):
    python scripts/build_qualtrics_json.py \\
        Final_Dataset/study_logs/experiment_20260419T154412Z.jsonl \\
        --round 1 \\
        --output Final_Dataset/Qualtrics/Round_1_Qualtrics_rebuilt.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pairs(path: Path, round_filter: int | None) -> list[dict]:
    """Pair SAE + AAVE trial_result records by prompt_index.

    When a prompt has multiple trials (N_TRIALS > 1) the first successful
    trial for each dialect is used, matching how Round 1 was built.
    """
    raw: dict[int, dict] = {}

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type") != "trial_result":
                continue
            if record.get("status") != "success":
                continue

            round_num = record.get("round")
            if round_filter is not None and round_num != round_filter:
                continue

            idx     = record["prompt_index"]
            dialect = record["dialect"]  # "SAE" or "AAVE"

            raw.setdefault(idx, {
                "prompt_index": idx,
                "category":     record.get("category", ""),
                "round":        round_num,
            })

            # Keep only the first successful trial per dialect per prompt
            key_prompt   = f"{dialect.lower()}_prompt"
            key_response = f"{dialect.lower()}_response"
            if key_response not in raw[idx]:
                raw[idx][key_prompt]   = record.get("prompt_text", "")
                raw[idx][key_response] = record.get("response_text") or ""

    pairs, skipped = [], 0
    for idx in sorted(raw):
        entry = raw[idx]
        required = ("sae_prompt", "aave_prompt", "sae_response", "aave_response")
        if all(k in entry for k in required):
            pairs.append(entry)
        else:
            missing = [k for k in required if k not in entry]
            print(f"  [warn] P{idx:02d}: missing {missing} — skipped")
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} incomplete prompt(s).")
    return pairs


def build_records(pairs: list[dict]) -> list[dict]:
    return [
        {
            "category":   p["category"],
            "prompt":     p["sae_prompt"],
            "response_A": p["sae_response"],
            "response_B": p["aave_response"],
            "label_A":    "SAE",
            "label_B":    "AAVE",
        }
        for p in pairs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", type=Path,
        help="Path to a study-log JSONL file.",
    )
    parser.add_argument(
        "--round", type=int, metavar="N",
        help="Only include records whose 'round' field equals N. "
             "Omit to include all rounds in the file.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSON path. Defaults to <input_stem>_qualtrics.json alongside the input.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    print(f"Loading: {args.input.name}")
    if args.round:
        print(f"  Filtering to round {args.round}")

    pairs = load_pairs(args.input, round_filter=args.round)
    if not pairs:
        raise SystemExit("No complete SAE+AAVE pairs found — nothing to write.")
    print(f"  {len(pairs)} pair(s) loaded")

    records = build_records(pairs)

    output = args.output or args.input.with_name(args.input.stem + "_qualtrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Wrote {output}")


if __name__ == "__main__":
    main()
