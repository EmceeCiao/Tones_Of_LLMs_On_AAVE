#!/usr/bin/env python3
"""
mitigation_jaccard.py

Computes token-level and bigram-level Jaccard similarity between SAE and AAVE
responses for a given mitigation condition JSONL.  No API calls required.

A low Jaccard score means the responses are genuinely different in surface form.
A score near 1.0 means the model collapsed to nearly identical outputs regardless
of dialect — which would undermine the study design.

Output:
  mitigation_study_logs/jaccard_<condition>__<timestamp>.csv   — per-pair scores
  mitigation_study_logs/jaccard_<condition>__<timestamp>.txt   — summary

Usage:
    # Analyse the default frozen set (ft_model_1_with_sysprompt, latest run):
    python mitigations/comparison/mitigation_jaccard.py

    # Analyse a specific file:
    python mitigations/comparison/mitigation_jaccard.py --input mitigations/comparison/mitigation_study_logs/ft_model_1_with_sysprompt__20260501T234707Z.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
LOGS_DIR     = SCRIPT_DIR / "mitigation_study_logs"
DEFAULT_FILE = LOGS_DIR / "ft_model_1_with_sysprompt__20260501T234707Z.jsonl"

HIGH_SIMILARITY_THRESHOLD = 0.80   # flag pairs above this as potentially collapsed


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> list[dict]:
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
            idx     = record["prompt_index"]
            dialect = record["dialect"]
            raw.setdefault(idx, {"prompt_index": idx, "category": record.get("category", "")})
            raw[idx][f"{dialect.lower()}_response"] = record["response_text"] or ""

    pairs = []
    for idx in sorted(raw):
        entry = raw[idx]
        if "sae_response" in entry and "aave_response" in entry:
            pairs.append(entry)
    return pairs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_pairs(pairs: list[dict]) -> list[dict]:
    rows = []
    for p in pairs:
        sae_tok  = tokenize(p["sae_response"])
        aave_tok = tokenize(p["aave_response"])

        sae_uni  = set(sae_tok)
        aave_uni = set(aave_tok)
        sae_bi   = set(bigrams(sae_tok))
        aave_bi  = set(bigrams(aave_tok))

        token_j  = jaccard(sae_uni, aave_uni)
        bigram_j = jaccard(sae_bi, aave_bi)

        rows.append({
            "prompt_index":      p["prompt_index"],
            "category":          p["category"],
            "token_jaccard":     round(token_j, 4),
            "bigram_jaccard":    round(bigram_j, 4),
            "sae_token_count":   len(sae_tok),
            "aave_token_count":  len(aave_tok),
            "flag_collapsed":    token_j >= HIGH_SIMILARITY_THRESHOLD,
        })
    return rows


def build_summary(rows: list[dict], source_file: str) -> str:
    tj = [r["token_jaccard"]  for r in rows]
    bj = [r["bigram_jaccard"] for r in rows]

    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["token_jaccard"])

    collapsed = [r for r in rows if r["flag_collapsed"]]

    lines = [
        "PHASE 7 — JACCARD SIMILARITY: ft_model_1_with_sysprompt (frozen set)",
        "=" * 62,
        f"Source file : {source_file}",
        f"Pairs       : {len(rows)}",
        f"Threshold   : token Jaccard ≥ {HIGH_SIMILARITY_THRESHOLD} flagged as collapsed",
        "",
        "TOKEN-LEVEL JACCARD (unigrams)",
        f"  Mean   : {statistics.mean(tj):.4f}",
        f"  Median : {statistics.median(tj):.4f}",
        f"  Min    : {min(tj):.4f}  (prompt_index {rows[tj.index(min(tj))]['prompt_index']})",
        f"  Max    : {max(tj):.4f}  (prompt_index {rows[tj.index(max(tj))]['prompt_index']})",
        f"  Stdev  : {statistics.stdev(tj):.4f}" if len(tj) > 1 else "",
        "",
        "BIGRAM-LEVEL JACCARD",
        f"  Mean   : {statistics.mean(bj):.4f}",
        f"  Median : {statistics.median(bj):.4f}",
        f"  Min    : {min(bj):.4f}",
        f"  Max    : {max(bj):.4f}",
        "",
        "BY CATEGORY (token Jaccard)",
    ]

    for cat in sorted(by_cat):
        scores = by_cat[cat]
        lines.append(
            f"  {cat:<12}  n={len(scores)}  "
            f"mean={statistics.mean(scores):.4f}  "
            f"median={statistics.median(scores):.4f}  "
            f"min={min(scores):.4f}  max={max(scores):.4f}"
        )

    lines += ["", "PER-PAIR SCORES", "-" * 40]
    for r in rows:
        flag = "  ← HIGH" if r["flag_collapsed"] else ""
        lines.append(
            f"  P{r['prompt_index']:02d}  [{r['category']:<10}]  "
            f"token={r['token_jaccard']:.4f}  bigram={r['bigram_jaccard']:.4f}{flag}"
        )

    lines += [""]
    if collapsed:
        lines.append(f"COLLAPSED PAIRS ({len(collapsed)}) — token Jaccard ≥ {HIGH_SIMILARITY_THRESHOLD}:")
        for r in collapsed:
            lines.append(f"  P{r['prompt_index']:02d}  [{r['category']}]  token={r['token_jaccard']:.4f}")
    else:
        lines.append(f"No collapsed pairs detected (all token Jaccard < {HIGH_SIMILARITY_THRESHOLD}).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Jaccard similarity check on ft_model_1_with_sysprompt responses.")
    parser.add_argument("--input", type=Path, default=DEFAULT_FILE,
                        help="Path to a phase7 condition JSONL file.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"File not found: {args.input}")

    pairs = load_pairs(args.input)
    if not pairs:
        raise SystemExit("No complete SAE+AAVE pairs found in file.")

    print(f"Loaded {len(pairs)} pairs from {args.input.name}")
    rows = analyse_pairs(pairs)

    run_id    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem      = re.sub(r"__\d{8}T\d{6}Z$", "", args.input.stem)
    csv_path  = LOGS_DIR / f"jaccard_{stem}__{run_id}.csv"
    txt_path  = LOGS_DIR / f"jaccard_{stem}__{run_id}.txt"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "prompt_index", "category",
            "token_jaccard", "bigram_jaccard",
            "sae_token_count", "aave_token_count", "flag_collapsed",
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows, args.input.name)
    txt_path.write_text(summary, encoding="utf-8")

    print(summary)
    print(f"\nCSV  → {csv_path}")
    print(f"Text → {txt_path}")


if __name__ == "__main__":
    main()
