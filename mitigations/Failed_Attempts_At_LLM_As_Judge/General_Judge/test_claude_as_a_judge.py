#!/usr/bin/env python3
"""Evaluate Claude as an abstaining LLM-as-judge for synthetic DPO labels.

This script reuses the judge11 prompt, scoring rule, response-order swapping,
metrics, and stable-DPO abstention logic. It calls Anthropic's Messages API
directly with Python standard-library HTTP tools, so no extra package is needed.

Set:
    export ANTHROPIC_API_KEY="..."

Run:
    ./.venv/bin/python dataset/test_claude_as_a_judge.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import test_single_llm_as_a_judge11 as base


DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1600
DEFAULT_TEMPERATURE = 0.0


def claude_message(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    anthropic_version: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": anthropic_version,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API HTTP {exc.code}: {body}") from exc


def output_text_from_claude_response(response: dict[str, Any]) -> str:
    pieces = []
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            pieces.append(str(block.get("text", "")))
    return "\n".join(piece for piece in pieces if piece).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Claude through the judge11 abstaining LLM-as-judge pipeline."
    )
    parser.add_argument("--input-path", type=Path, default=base.DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=base.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--anthropic-version", default=DEFAULT_ANTHROPIC_VERSION)
    parser.add_argument("--n-trials", type=int, default=base.N_TRIALS)
    parser.add_argument(
        "--min-score-margin",
        type=int,
        default=2,
        help="Minimum score-total gap required in both response orders before keeping a DPO label.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the gold prompts and print planned Claude calls without calling Anthropic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_items = base.load_gold_items(args.input_path)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_path": str(args.input_path),
                    "num_gold_items": len(gold_items),
                    "model": args.model,
                    "num_orderings": len(base.ORDERINGS),
                    "num_eval_cases": len(gold_items) * len(base.ORDERINGS) * args.n_trials,
                    "min_score_margin": args.min_score_margin,
                    "orderings": base.ORDERINGS,
                    "gold_items": [
                        {
                            "prompt_id": item["prompt_id"],
                            "category": item["category"],
                            "gold_label": item["gold_label"],
                        }
                        for item in gold_items
                    ],
                },
                indent=2,
            )
        )
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.output_dir / f"claude_judge_{run_id}.jsonl"
    summary_path = args.output_dir / f"claude_judge_{run_id}_summary.json"

    experiment_config = {
        "run_id": run_id,
        "provider": "anthropic",
        "model": args.model,
        "input_path": str(args.input_path),
        "gold_labels": base.GOLD_LABELS,
        "num_gold_items": len(gold_items),
        "n_trials": args.n_trials,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "anthropic_version": args.anthropic_version,
        "min_score_margin": args.min_score_margin,
        "judge_prompt_type": "claude_v1_abstaining_few_shot_score_derived_dpo_filter",
        "score_fields": base.SCORE_FIELDS,
        "orderings": base.ORDERINGS,
    }

    base.append_jsonl(
        log_path,
        {
            "record_type": "experiment_config",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": experiment_config,
        },
    )

    results = []
    for item in gold_items:
        for trial in range(1, args.n_trials + 1):
            for ordering in base.ORDERINGS:
                trial_start = time.time()
                prompt = base.build_judge_prompt(item, ordering)
                order_name = ordering["order_name"]
                base_record = {
                    "record_type": "trial_result",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "config": experiment_config,
                    "prompt_id": item["prompt_id"],
                    "category": item["category"],
                    "gold_label": item["gold_label"],
                    "trial": trial,
                    "model": args.model,
                    "provider": "anthropic",
                    "order_name": order_name,
                    "response_a_label": ordering["response_a_label"],
                    "response_b_label": ordering["response_b_label"],
                }

                try:
                    claude_response = claude_message(
                        api_key=api_key,
                        model=args.model,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        anthropic_version=args.anthropic_version,
                    )
                    response_text = output_text_from_claude_response(claude_response)
                    judge_result, parse_error = base.parse_judge_output(response_text)
                    response_choice, derivation = base.derive_response_choice_from_scores(judge_result)
                    predicted_label = base.response_choice_to_label(response_choice, ordering)
                    result_row = {
                        **base_record,
                        "status": "success" if parse_error is None else "parse_error",
                        "response_id": claude_response.get("id"),
                        "response_text": response_text,
                        "claude_response": claude_response,
                        "parse_error": parse_error,
                        "judge_result": judge_result,
                        "score_derivation": derivation,
                        "response_choice": response_choice,
                        "predicted_label": predicted_label,
                        "correct": predicted_label == item["gold_label"],
                        "duration_sec": time.time() - trial_start,
                    }
                    results.append(result_row)
                    base.append_jsonl(log_path, result_row)
                    print(
                        f"Saved {item['prompt_id']} | model={args.model} | order={order_name} "
                        f"| gold={item['gold_label']} | predicted={predicted_label} "
                        f"| trial {trial}"
                    )

                except Exception as exc:
                    result_row = {
                        **base_record,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "response_choice": base.INVALID_LABEL,
                        "predicted_label": base.INVALID_LABEL,
                        "correct": False,
                        "duration_sec": time.time() - trial_start,
                    }
                    results.append(result_row)
                    base.append_jsonl(log_path, result_row)
                    print(f"{item['prompt_id']} | model={args.model} | order={order_name} | trial {trial} failed and was logged.")

    metrics = base.compute_metrics(results)
    model_metrics = base.compute_model_metrics(results)
    order_bias_summary = base.compute_order_bias_summary(results)
    majority_vote_summary = base.compute_majority_vote_summary(results)
    stable_dpo_decisions = base.build_stable_dpo_decisions(results, args.min_score_margin)
    stable_dpo_summary = base.compute_stable_dpo_summary(stable_dpo_decisions)

    summary = {
        "config": experiment_config,
        "metrics": metrics,
        "model_metrics": model_metrics,
        "order_bias_summary": order_bias_summary,
        "majority_vote_summary": majority_vote_summary,
        "stable_dpo_summary": stable_dpo_summary,
        "results": [
            {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "trial": row["trial"],
                "model": row["model"],
                "provider": row["provider"],
                "order_name": row["order_name"],
                "response_a_label": row["response_a_label"],
                "response_b_label": row["response_b_label"],
                "gold_label": row["gold_label"],
                "response_choice": row.get("response_choice"),
                "predicted_label": row["predicted_label"],
                "correct": row["correct"],
                "status": row["status"],
                "parse_error": row.get("parse_error"),
                "judge_result": row.get("judge_result"),
                "score_derivation": row.get("score_derivation"),
            }
            for row in results
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    base.append_jsonl(
        log_path,
        {
            "record_type": "experiment_end",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": experiment_config,
            "summary_path": str(summary_path),
            "metrics": metrics,
            "model_metrics": model_metrics,
            "order_bias_summary": order_bias_summary,
            "majority_vote_summary": majority_vote_summary,
            "stable_dpo_summary": stable_dpo_summary,
        },
    )

    base.print_run_summary(metrics, model_metrics, majority_vote_summary, order_bias_summary)
    base.print_stable_dpo_summary(stable_dpo_summary)
    print(f"Done. JSONL results saved to: {log_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
