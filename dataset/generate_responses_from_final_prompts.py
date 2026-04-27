#!/usr/bin/env python3

import json
import os
import time
from datetime import datetime, timezone
from openai import OpenAI

client = OpenAI()

INPUT_JSON_PATH = "final_prompts.json"
ROUND_SCOPE = "round 1"   # "round 1", "round 2", "leftover", or "all"

MODEL_NAME = "gpt-4.1"
N_TRIALS = 1   # set to 5 if you want repeated runs per prompt variant
SEED = 12345
TEMPERATURE = 1
TOP_P = 1

# This is still just metadata unless you explicitly add a reasoning control
# that your model/SDK supports in the request itself.
REASONING_TYPE = "medium"

OUTPUT_DIR = "study_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
log_path = os.path.join(OUTPUT_DIR, f"experiment_{run_id}.jsonl")


def load_prompt_items(input_json_path: str, round_scope: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("Prompt file must be either a list or a grouped dict.")

    sections = ["round 1", "round 2", "leftover"] if round_scope == "all" else [round_scope]

    prompt_items = []
    for section in sections:
        for item in data.get(section, []):
            new_item = dict(item)
            if section == "round 1":
                new_item["round"] = 1
            elif section == "round 2":
                new_item["round"] = 2
            else:
                new_item["round"] = None
            prompt_items.append(new_item)

    return prompt_items


prompt_items = load_prompt_items(INPUT_JSON_PATH, ROUND_SCOPE)

experiment_config = {
    "run_id": run_id,
    "model": MODEL_NAME,
    "input_json_path": INPUT_JSON_PATH,
    "round_scope": ROUND_SCOPE,
    "num_prompt_items": len(prompt_items),
    "n_trials_per_prompt_variant": N_TRIALS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    # "seed": SEED,
    # "reasoning_type": REASONING_TYPE
}


def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


append_jsonl(log_path, {
    "record_type": "experiment_config",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "config": experiment_config
})

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
                response = None

                if MODEL_NAME == "gpt-5.4":
                    response = client.responses.create(
                        model=MODEL_NAME,
                        input=prompt_text,
                        # temperature=TEMPERATURE,
                        # top_p=TOP_P,
                        # reasoning={"effort": REASONING_TYPE, "summary":"detailed"},
                    )
                    seed_used_in_request = False
                else:
                    response = client.responses.create(
                        model=MODEL_NAME,
                        input=prompt_text,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        # seed=SEED,
                    )
                    seed_used_in_request = False

                print("Entered Try\n")

                reasoning_texts = []

                for output_item in getattr(response, "output", []) or []:
                    if getattr(output_item, "type", None) == "reasoning":
                        for s in getattr(output_item, "summary", []) or []:
                            text = getattr(s, "text", None)
                            if text:
                                reasoning_texts.append(text)

                record = {
                    "record_type": "trial_result",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "config": experiment_config,

                    "prompt_index": prompt_index,
                    "category": category,
                    "round": round_num,
                    "dialect": dialect_label,
                    "prompt_text": prompt_text,

                    "trial": trial,
                    "status": "success",

                    "response_id": getattr(response, "id", None),
                    "response_text": getattr(response, "output_text", None),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),

                    "reasoning_type": REASONING_TYPE,
                    "reasoning_text": "\n".join(reasoning_texts) if reasoning_texts else None,

                    "seed_used_in_request": seed_used_in_request,
                    "duration_sec": time.time() - trial_start
                }

                append_jsonl(log_path, record)
                print(f"Saved prompt {prompt_index} | {dialect_label} | trial {trial}")

            except Exception as e:
                append_jsonl(log_path, {
                    "record_type": "trial_result",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "config": experiment_config,

                    "prompt_index": prompt_index,
                    "category": category,
                    "round": round_num,
                    "dialect": dialect_label,
                    "prompt_text": prompt_text,

                    "trial": trial,
                    "status": "error",

                    "reasoning_type": REASONING_TYPE,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_sec": time.time() - trial_start
                })

                print(f"Prompt {prompt_index} | {dialect_label} | trial {trial} failed and was logged.")

append_jsonl(log_path, {
    "record_type": "experiment_end",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "config": experiment_config
})

print(f"Done. Results saved to: {log_path}")
