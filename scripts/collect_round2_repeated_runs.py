#!/usr/bin/env python3
"""
collect_round2_repeated_runs.py

Runs the fine-tuned model (ft:gpt-4.1-2025-04-14:personal:aave-dpo-4:DaYoNkrP)
with the mitigation system prompt 5 times on all 5 Round 2 prompts (both SAE
and AAVE variants).  Mirrors the setup used in generate_responses_round2.py
exactly, adding only the repetition needed to separate dialect-driven
differences from temperature noise.

Each trial is numbered 1–5.  All 50 results (5 prompts × 2 dialects × 5
trials) are streamed to a single JSONL as they complete.

Output:
    Final_Dataset/study_logs/round2_repeated_runs.jsonl

Usage:
    export OPENAI_API_KEY="your-key"
    python scripts/collect_round2_repeated_runs.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

REPO_ROOT          = Path(__file__).resolve().parent.parent
PROMPTS_FILE       = REPO_ROOT / "Final_Dataset" / "final_prompts.json"
SYSTEM_PROMPT_FILE = REPO_ROOT / "mitigations" / "system_prompt.txt"
OUTPUT_PATH        = REPO_ROOT / "Final_Dataset" / "study_logs" / "round2_repeated_runs.jsonl"

MODEL       = "ft:gpt-4.1-2025-04-14:personal:aave-dpo-4:DaYoNkrP"
TEMPERATURE = 1
TOP_P       = 1
N_TRIALS    = 5


def load_prompts(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for i, item in enumerate(data["round 2"], start=1):
        items.append({
            "prompt_index": i,
            "category":     item["category"],
            "sae_prompt":   item["sae_prompt"],
            "aave_prompt":  item["aave_prompt"],
        })
    return items


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: OPENAI_API_KEY environment variable is not set.")

    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"Output file already exists: {OUTPUT_PATH}\n"
            "Delete or rename it before re-running to avoid duplicate data."
        )

    system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    client  = OpenAI(api_key=api_key)
    prompts = load_prompts(PROMPTS_FILE)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = {
        "run_id":            run_id,
        "model":             MODEL,
        "round_scope":       "round 2",
        "system_prompt":     system_prompt,
        "temperature":       TEMPERATURE,
        "top_p":             TOP_P,
        "n_trials":          N_TRIALS,
        "num_prompts":       len(prompts),
        "use_system_prompt": True,
    }
    append_jsonl(OUTPUT_PATH, {
        "record_type":   "experiment_config",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config":        config,
    })

    total = len(prompts) * 2 * N_TRIALS
    done  = 0

    for trial in range(1, N_TRIALS + 1):
        print(f"\n── Trial {trial}/{N_TRIALS} ──────────────────────────────")
        for item in prompts:
            for dialect_key, dialect_label in [("sae_prompt", "SAE"), ("aave_prompt", "AAVE")]:
                prompt_text = item[dialect_key]
                start = time.time()
                try:
                    response = client.responses.create(
                        model=MODEL,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": prompt_text},
                        ],
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )
                    append_jsonl(OUTPUT_PATH, {
                        "record_type":        "trial_result",
                        "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
                        "config":             config,
                        "prompt_index":       item["prompt_index"],
                        "category":           item["category"],
                        "dialect":            dialect_label,
                        "prompt_text":        prompt_text,
                        "trial":              trial,
                        "status":             "success",
                        "response_id":        getattr(response, "id", None),
                        "response_text":      getattr(response, "output_text", None),
                        "system_fingerprint": getattr(response, "system_fingerprint", None),
                        "duration_sec":       round(time.time() - start, 3),
                    })
                except Exception as e:
                    append_jsonl(OUTPUT_PATH, {
                        "record_type":   "trial_result",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "config":        config,
                        "prompt_index":  item["prompt_index"],
                        "category":      item["category"],
                        "dialect":       dialect_label,
                        "prompt_text":   prompt_text,
                        "trial":         trial,
                        "status":        "error",
                        "error_type":    type(e).__name__,
                        "error_message": str(e),
                        "duration_sec":  round(time.time() - start, 3),
                    })
                    print(f"  ERROR P{item['prompt_index']:02d} {dialect_label} trial {trial}: {e}")

                done += 1
                print(f"  P{item['prompt_index']:02d} [{item['category']:<10}] {dialect_label} trial {trial} — saved  ({done}/{total})")

    append_jsonl(OUTPUT_PATH, {
        "record_type":   "experiment_end",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config":        config,
    })
    print(f"\nDone. {done} responses saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
