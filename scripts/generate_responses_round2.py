#!/usr/bin/env python3
"""
generate_responses_round2.py

Generates Round 2 responses using FT Model 1 + system prompt.
Mirrors the Round 1 generation script structure; results stream
to study_logs_round2/ as they arrive.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/generate_responses_round2.py
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT          = Path(__file__).resolve().parent.parent
PROMPTS_FILE       = REPO_ROOT / "Final_Dataset" / "final_prompts.json"
SYSTEM_PROMPT_FILE = REPO_ROOT / "mitigations" / "system_prompt.txt"
OUTPUT_DIR         = REPO_ROOT / "study_logs_round2"

MODEL_NAME  = "ft:gpt-4.1-2025-04-14:personal:aave-dpo-4:DaYoNkrP"
TEMPERATURE = 1
TOP_P       = 1
N_TRIALS    = 1

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
client = OpenAI()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
run_id   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
log_path = OUTPUT_DIR / f"round2_{run_id}.jsonl"

system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

with PROMPTS_FILE.open(encoding="utf-8") as f:
    all_prompts = json.load(f)

prompt_items = [
    {**item, "round": 2}
    for item in all_prompts.get("round 2", [])
]

experiment_config = {
    "run_id":                   run_id,
    "model":                    MODEL_NAME,
    "round_scope":              "round 2",
    "system_prompt":            system_prompt,
    "num_prompt_items":         len(prompt_items),
    "n_trials_per_prompt_variant": N_TRIALS,
    "temperature":              TEMPERATURE,
    "top_p":                    TOP_P,
}


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


append_jsonl(log_path, {
    "record_type":   "experiment_config",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "config":        experiment_config,
})

print(f"Round 2 generation — {len(prompt_items)} prompts × 2 dialects × {N_TRIALS} trials")
print(f"Model : {MODEL_NAME}")
print(f"Log   : {log_path}\n")

for prompt_index, item in enumerate(prompt_items, start=1):
    category  = item.get("category")
    round_num = item.get("round")

    for dialect_key, dialect_label in [("sae_prompt", "SAE"), ("aave_prompt", "AAVE")]:
        prompt_text = item.get(dialect_key)
        if not prompt_text:
            continue

        for trial in range(1, N_TRIALS + 1):
            trial_start = time.time()
            try:
                response = client.responses.create(
                    model=MODEL_NAME,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt_text},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )

                append_jsonl(log_path, {
                    "record_type":        "trial_result",
                    "timestamp_utc":      datetime.now(timezone.utc).isoformat(),
                    "config":             experiment_config,
                    "prompt_index":       prompt_index,
                    "category":           category,
                    "round":              round_num,
                    "dialect":            dialect_label,
                    "prompt_text":        prompt_text,
                    "trial":              trial,
                    "status":             "success",
                    "response_id":        getattr(response, "id", None),
                    "response_text":      getattr(response, "output_text", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                    "duration_sec":       time.time() - trial_start,
                })
                print(f"  Saved P{prompt_index:02d} | {dialect_label} | trial {trial}")

            except Exception as e:
                append_jsonl(log_path, {
                    "record_type":   "trial_result",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "config":        experiment_config,
                    "prompt_index":  prompt_index,
                    "category":      category,
                    "round":         round_num,
                    "dialect":       dialect_label,
                    "prompt_text":   prompt_text,
                    "trial":         trial,
                    "status":        "error",
                    "error_type":    type(e).__name__,
                    "error_message": str(e),
                    "duration_sec":  time.time() - trial_start,
                })
                print(f"  ERROR P{prompt_index:02d} | {dialect_label} | trial {trial}: {e}")

append_jsonl(log_path, {
    "record_type":   "experiment_end",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "config":        experiment_config,
})

print(f"\nDone. Results saved to: {log_path}")
