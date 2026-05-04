#!/usr/bin/env python3
"""
mitigation_check.py

Runs 6 experiment conditions over all Round 1 prompts (30 variants: 15 SAE + 15 AAVE).
Each condition streams results to its own JSONL file under mitigation_study_logs/.

Conditions:
  1. ft_model_1_only          — FT Model 1, no system prompt
  2. ft_model_1_with_sysprompt — FT Model 1 + system prompt
  3. ft_model_2_only          — FT Model 2, no system prompt
  4. ft_model_2_with_sysprompt — FT Model 2 + system prompt
  5. base_sysprompt_run1      — Base GPT-4.1 + system prompt (run 1)
  6. base_sysprompt_run2      — Base GPT-4.1 + system prompt (run 2)

Usage:
    # Run all 6 conditions:
    python mitigations/comparison/mitigation_check.py

    # Run specific conditions only (by name or 1-indexed number):
    python mitigations/comparison/mitigation_check.py --conditions 1 3 5
    python mitigations/comparison/mitigation_check.py --conditions ft_model_1_only base_sysprompt_run1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR      = Path(__file__).resolve().parent
REPO_ROOT       = SCRIPT_DIR.parent.parent
PROMPTS_FILE    = REPO_ROOT / "Final_Dataset" / "final_prompts.json"
SYSTEM_PROMPT_FILE = REPO_ROOT / "mitigations" / "system_prompt.txt"
OUTPUT_DIR      = SCRIPT_DIR / "mitigation_study_logs"

# ---------------------------------------------------------------------------
# Fill in your fine-tuned model IDs once Phase 6 is complete.
# ---------------------------------------------------------------------------
FT_MODEL_1 = "ft:gpt-4.1-2025-04-14:personal:aave-dpo-4:DaYoNkrP"   # e.g. "ft:gpt-4.1-2025-04-14:org:name:abc123"
FT_MODEL_2 = "ft:gpt-4.1-2025-04-14:personal:aave-dpo-2:DaZ4HKta"   # e.g. "ft:gpt-4.1-2025-04-14:org:name:def456"
BASE_MODEL  = "gpt-4.1-2025-04-14"

TEMPERATURE = 1
TOP_P       = 1
N_TRIALS    = 1  # set higher if you want repeated samples per condition

# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------
CONDITIONS = [
    {
        "name":              "ft_model_1_only",
        "model":             FT_MODEL_1,
        "use_system_prompt": False,
    },
    {
        "name":              "ft_model_1_with_sysprompt",
        "model":             FT_MODEL_1,
        "use_system_prompt": True,
    },
    {
        "name":              "ft_model_2_only",
        "model":             FT_MODEL_2,
        "use_system_prompt": False,
    },
    {
        "name":              "ft_model_2_with_sysprompt",
        "model":             FT_MODEL_2,
        "use_system_prompt": True,
    },
    {
        "name":              "base_sysprompt_run1",
        "model":             BASE_MODEL,
        "use_system_prompt": True,
    },
    {
        "name":              "base_sysprompt_run2",
        "model":             BASE_MODEL,
        "use_system_prompt": True,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_round1_prompts(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for item in data.get("round 1", []):
        entry = dict(item)
        entry["round"] = 1
        items.append(entry)
    return items


def load_system_prompt(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        return f.read().strip()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def parse_condition_args(raw: list[str]) -> list[dict]:
    """Resolve --conditions arguments (names or 1-based indices) to condition dicts."""
    selected = []
    for token in raw:
        if token.isdigit():
            idx = int(token) - 1
            if not 0 <= idx < len(CONDITIONS):
                raise SystemExit(f"Condition index {token} out of range (1–{len(CONDITIONS)})")
            selected.append(CONDITIONS[idx])
        else:
            match = next((c for c in CONDITIONS if c["name"] == token), None)
            if match is None:
                names = ", ".join(c["name"] for c in CONDITIONS)
                raise SystemExit(f"Unknown condition '{token}'. Valid names: {names}")
            selected.append(match)
    return selected


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_condition(client: OpenAI, condition: dict, prompt_items: list[dict],
                  system_prompt: str, output_dir: Path) -> None:
    cond_name = condition["name"]
    model     = condition["model"]
    use_sysp  = condition["use_system_prompt"]

    if "PLACEHOLDER" in model:
        print(f"\n[SKIP] {cond_name}: model ID is still a placeholder — update FT_MODEL_1 / FT_MODEL_2 in this script.")
        return

    run_id   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = output_dir / f"{cond_name}__{run_id}.jsonl"

    experiment_config = {
        "run_id":             run_id,
        "condition_name":     cond_name,
        "model":              model,
        "use_system_prompt":  use_sysp,
        "temperature":        TEMPERATURE,
        "top_p":              TOP_P,
        "n_trials":           N_TRIALS,
        "num_prompt_items":   len(prompt_items),
    }

    append_jsonl(log_path, {
        "record_type":   "experiment_config",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config":        experiment_config,
    })

    print(f"\n{'='*60}")
    print(f"Condition : {cond_name}")
    print(f"Model     : {model}")
    print(f"Sys prompt: {use_sysp}")
    print(f"Log       : {log_path.name}")
    print(f"{'='*60}")

    for prompt_index, item in enumerate(prompt_items, start=1):
        category = item.get("category")
        round_num = item.get("round")

        for dialect_key, dialect_label in [("sae_prompt", "SAE"), ("aave_prompt", "AAVE")]:
            prompt_text = item.get(dialect_key)
            if not prompt_text:
                continue

            for trial in range(1, N_TRIALS + 1):
                trial_start = time.time()
                try:
                    if use_sysp:
                        input_messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": prompt_text},
                        ]
                    else:
                        input_messages = [
                            {"role": "user", "content": prompt_text},
                        ]

                    response = client.responses.create(
                        model=model,
                        input=input_messages,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )

                    record = {
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
                    }
                    append_jsonl(log_path, record)
                    print(f"  [{cond_name}] prompt {prompt_index:02d} | {dialect_label} | trial {trial} — saved")

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
                    print(f"  [{cond_name}] prompt {prompt_index:02d} | {dialect_label} | trial {trial} — ERROR: {e}")

    append_jsonl(log_path, {
        "record_type":   "experiment_end",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config":        experiment_config,
    })
    print(f"\nCondition '{cond_name}' complete → {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 regression check — 6-condition experiment runner.")
    parser.add_argument(
        "--conditions", nargs="+", metavar="COND",
        help="Run only these conditions (names or 1-based indices). Omit to run all 6.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: OPENAI_API_KEY environment variable is not set.")

    client        = OpenAI(api_key=api_key)
    prompt_items  = load_round1_prompts(PROMPTS_FILE)
    system_prompt = load_system_prompt(SYSTEM_PROMPT_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conditions_to_run = parse_condition_args(args.conditions) if args.conditions else CONDITIONS

    print(f"Phase 7 — running {len(conditions_to_run)} condition(s) over {len(prompt_items)} prompt items "
          f"({len(prompt_items) * 2} variants × {N_TRIALS} trial(s) each).")

    for condition in conditions_to_run:
        run_condition(client, condition, prompt_items, system_prompt, OUTPUT_DIR)

    print("\nAll requested conditions complete.")


if __name__ == "__main__":
    main()
