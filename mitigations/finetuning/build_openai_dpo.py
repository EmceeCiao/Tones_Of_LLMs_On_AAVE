#!/usr/bin/env python3
"""
build_openai_dpo.py

Merges qualtrics_dpo_pairs.jsonl and user_attribution_dpo_pairs.jsonl into a
single JSONL file formatted for the OpenAI fine-tuning dashboard (DPO).

OpenAI DPO format (one JSON object per line):
{
  "input": {
    "messages": [
      {"role": "system", "content": "<system prompt>"},
      {"role": "user",   "content": "<prompt>"}
    ]
  },
  "preferred_output":     [{"role": "assistant", "content": "<chosen response>"}],
  "non_preferred_output": [{"role": "assistant", "content": "<rejected response>"}]
}

Sources combined:
  qualtrics_dpo_pairs.jsonl      — human preference pairs (strong Qualtrics majority)
  user_attribution_dpo_pairs.jsonl — LLM-judge pairs (confirmed user-attribution drift)

The system prompt is loaded from mitigations/system_prompt.txt so both this script
and the Round 2 response generation share the exact same prompt text.

Usage:
    python mitigations/finetuning/build_openai_dpo.py

Output:
    mitigations/finetuning/openai_dpo_train.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent

SYSTEM_PROMPT_FILE       = REPO_ROOT / "mitigations" / "system_prompt.txt"
QUALTRICS_JSONL          = SCRIPT_DIR / "qualtrics_dpo_pairs.jsonl"
USER_ATTRIBUTION_JSONL   = SCRIPT_DIR / "user_attribution_dpo_pairs.jsonl"
OUTPUT_JSONL             = SCRIPT_DIR / "openai_dpo_train.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_openai_dpo(record: dict, system_prompt: str) -> dict:
    return {
        "input": {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": record["prompt"]},
            ]
        },
        "preferred_output":     [{"role": "assistant", "content": record["chosen"]}],
        "non_preferred_output": [{"role": "assistant", "content": record["rejected"]}],
    }


def main() -> None:
    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    print(f"System prompt loaded from {SYSTEM_PROMPT_FILE.name} ({len(system_prompt)} chars)")

    qualtrics        = load_jsonl(QUALTRICS_JSONL)
    user_attribution = load_jsonl(USER_ATTRIBUTION_JSONL)

    print(f"  qualtrics_dpo_pairs.jsonl      : {len(qualtrics)} records")
    print(f"  user_attribution_dpo_pairs.jsonl: {len(user_attribution)} records")

    all_records = qualtrics + user_attribution
    print(f"  Total                          : {len(all_records)} records")

    output = [to_openai_dpo(r, system_prompt) for r in all_records]

    with OUTPUT_JSONL.open("w", encoding="utf-8") as fh:
        for obj in output:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(output)} records → {OUTPUT_JSONL}")

    # Breakdown by source for traceability
    q_versions  = {}
    ua_versions = {}
    for r in qualtrics:
        k = r.get("prompt_version", "?")
        q_versions[k] = q_versions.get(k, 0) + 1
    for r in user_attribution:
        k = r.get("prompt_version", "?")
        ua_versions[k] = ua_versions.get(k, 0) + 1

    print()
    print("Breakdown:")
    print(f"  Qualtrics  — AAVE-prompt: {q_versions.get('AAVE', 0)}  SAE-prompt: {q_versions.get('SAE', 0)}")
    print(f"  User-attr  — AAVE-prompt: {ua_versions.get('AAVE', 0)}  SAE-prompt: {ua_versions.get('SAE', 0)}")

    # Spot-check: print first record so format is easy to verify
    print()
    print("First record (spot-check):")
    first = output[0]
    print(f"  system  : {first['input']['messages'][0]['content'][:80]}…")
    print(f"  user    : {first['input']['messages'][1]['content'][:80]}…")
    print(f"  chosen  : {first['preferred_output'][0]['content'][:80]}…")
    print(f"  rejected: {first['non_preferred_output'][0]['content'][:80]}…")


if __name__ == "__main__":
    main()
