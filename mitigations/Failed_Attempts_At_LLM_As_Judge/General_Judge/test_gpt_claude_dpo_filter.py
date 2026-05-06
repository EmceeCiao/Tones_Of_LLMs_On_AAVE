#!/usr/bin/env python3
"""Cross-provider abstaining DPO-label filter using GPT + Claude.

This script is for creating *stable synthetic DPO labels*, not for forcing a
label on every pair.

For each gold/evaluation pair, it:
1. Shows only the SAE/source prompt and both candidate responses.
2. Runs each provider with both response orders.
3. Converts dimension scores into SAE/AAVE/No Preference.
4. Builds a provider-level stable decision only when both response orders agree.
5. Keeps a synthetic DPO label only when GPT and Claude both keep the same
   binary label with sufficient score margins.

Set:
    export OPENAI_API_KEY="..."
    export ANTHROPIC_API_KEY="..."

Run:
    ./.venv/bin/python dataset/test_gpt_claude_dpo_filter.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

import test_claude_as_a_judge as claude_judge
import test_single_llm_as_a_judge11 as base


DEFAULT_GPT_MODEL = "gpt-4.1"
DEFAULT_CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", claude_judge.DEFAULT_MODEL)
DEFAULT_OUTPUT_DIR = base.DEFAULT_OUTPUT_DIR


def create_gpt_response(client: OpenAI, model: str, prompt: str, temperature: float, top_p: float):
    if model.startswith("gpt-5"):
        return client.responses.create(model=model, input=prompt)
    return client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
    )


def provider_record_base(
    *,
    provider: str,
    model: str,
    item: dict[str, Any],
    ordering: dict[str, str],
    trial: int,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "trial_result",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": experiment_config,
        "provider": provider,
        "model": model,
        "prompt_id": item["prompt_id"],
        "category": item["category"],
        "gold_label": item["gold_label"],
        "trial": trial,
        "order_name": ordering["order_name"],
        "response_a_label": ordering["response_a_label"],
        "response_b_label": ordering["response_b_label"],
    }


def parse_score_result(
    *,
    base_record: dict[str, Any],
    response_text: str | None,
    response_id: str | None,
    raw_response: Any,
    duration_sec: float,
) -> dict[str, Any]:
    judge_result, parse_error = base.parse_judge_output(response_text)
    response_choice, derivation = base.derive_response_choice_from_scores(judge_result)
    ordering = {
        "response_a_label": base_record["response_a_label"],
        "response_b_label": base_record["response_b_label"],
    }
    predicted_label = base.response_choice_to_label(response_choice, ordering)
    return {
        **base_record,
        "status": "success" if parse_error is None else "parse_error",
        "response_id": response_id,
        "response_text": response_text,
        "raw_response": raw_response,
        "parse_error": parse_error,
        "judge_result": judge_result,
        "score_derivation": derivation,
        "response_choice": response_choice,
        "predicted_label": predicted_label,
        "correct": predicted_label == base_record["gold_label"],
        "duration_sec": duration_sec,
    }


def build_provider_decisions(results: list[dict[str, Any]], min_score_margin: int) -> list[dict[str, Any]]:
    decisions = []
    providers = sorted({row["provider"] for row in results})
    for provider in providers:
        provider_results = [row for row in results if row["provider"] == provider]
        decisions.extend(base.build_stable_dpo_decisions(provider_results, min_score_margin))
    for decision in decisions:
        matching_rows = [
            row for row in results
            if row["prompt_id"] == decision["prompt_id"]
            and row["trial"] == decision["trial"]
            and row["model"] == decision["model"]
        ]
        decision["provider"] = matching_rows[0]["provider"] if matching_rows else ""
    return decisions


def build_cross_provider_decisions(provider_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in provider_decisions:
        grouped.setdefault((row["prompt_id"], row["trial"]), []).append(row)

    cross_decisions = []
    for (prompt_id, trial), rows in sorted(grouped.items()):
        first_row = rows[0]
        by_provider = {row["provider"]: row for row in rows}
        kept_rows = [row for row in rows if row["kept_for_dpo"]]
        labels = [row["predicted_label"] for row in kept_rows]

        decision = {
            "prompt_id": prompt_id,
            "category": first_row["category"],
            "trial": trial,
            "gold_label": first_row["gold_label"],
            "kept_for_dpo": False,
            "predicted_label": None,
            "chosen_label": None,
            "rejected_label": None,
            "correct": False,
            "abstain_reason": "",
            "provider_decisions": by_provider,
        }

        required_providers = {"openai", "anthropic"}
        if set(by_provider) != required_providers:
            decision["abstain_reason"] = "missing_provider_decision"
        elif any(not row["kept_for_dpo"] for row in rows):
            reasons = {
                f"{row['provider']}:{row['abstain_reason'] or 'abstained'}"
                for row in rows
                if not row["kept_for_dpo"]
            }
            decision["abstain_reason"] = ";".join(sorted(reasons))
        elif len(set(labels)) != 1:
            decision["abstain_reason"] = "provider_disagreement"
        elif labels[0] not in {"SAE", "AAVE"}:
            decision["abstain_reason"] = "not_binary_preference"
        else:
            label = labels[0]
            decision["kept_for_dpo"] = True
            decision["predicted_label"] = label
            decision["chosen_label"] = label
            decision["rejected_label"] = "AAVE" if label == "SAE" else "SAE"
            decision["correct"] = label == first_row["gold_label"]

        cross_decisions.append(decision)
    return cross_decisions


def summarize_cross_provider_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    kept = [row for row in decisions if row["kept_for_dpo"]]
    abstained = [row for row in decisions if not row["kept_for_dpo"]]
    abstain_reasons: dict[str, int] = {}
    for row in abstained:
        reason = row["abstain_reason"] or "unknown"
        abstain_reasons[reason] = abstain_reasons.get(reason, 0) + 1

    metric_rows = [
        {"gold_label": row["gold_label"], "predicted_label": row["predicted_label"]}
        for row in kept
    ]
    return {
        "total_prompt_trials": len(decisions),
        "kept_count": len(kept),
        "abstained_count": len(abstained),
        "coverage": len(kept) / len(decisions) if decisions else None,
        "kept_accuracy": (
            sum(1 for row in kept if row["correct"]) / len(kept)
            if kept
            else None
        ),
        "kept_metrics": base.compute_metrics(metric_rows) if metric_rows else None,
        "abstain_reasons": abstain_reasons,
        "decisions": decisions,
    }


def print_cross_provider_summary(summary: dict[str, Any]) -> None:
    print("\n=== Cross-Provider Stable DPO Summary ===")
    print(
        "  kept: "
        f"{summary['kept_count']}/{summary['total_prompt_trials']} "
        f"(coverage={base.fmt_rate(summary['coverage'])})"
    )
    print(f"  kept accuracy vs gold: {base.fmt_rate(summary['kept_accuracy'])}")
    print(f"  abstained: {summary['abstained_count']}")
    if summary["abstain_reasons"]:
        print("  abstain reasons:")
        for reason, count in sorted(summary["abstain_reasons"].items()):
            print(f"    {reason}: {count}")
    if summary.get("kept_metrics"):
        base.print_metrics_block("Cross-provider kept DPO labels only", summary["kept_metrics"])
    for row in summary["decisions"]:
        status = "KEEP" if row["kept_for_dpo"] else "ABSTAIN"
        detail = (
            f"chosen={row['chosen_label']} rejected={row['rejected_label']}"
            if row["kept_for_dpo"]
            else f"reason={row['abstain_reason']}"
        )
        provider_labels = {
            provider: decision.get("predicted_label")
            for provider, decision in row["provider_decisions"].items()
        }
        print(
            f"  {status} {row['prompt_id']} gold={row['gold_label']} "
            f"{detail} provider_labels={provider_labels}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT + Claude and keep only cross-provider stable DPO labels."
    )
    parser.add_argument("--input-path", type=Path, default=base.DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpt-model", default=DEFAULT_GPT_MODEL)
    parser.add_argument("--claude-model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--claude-max-tokens", type=int, default=claude_judge.DEFAULT_MAX_TOKENS)
    parser.add_argument("--claude-version", default=claude_judge.DEFAULT_ANTHROPIC_VERSION)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--min-score-margin", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned calls without calling either provider.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_items = base.load_gold_items(args.input_path)

    planned_calls = len(gold_items) * len(base.ORDERINGS) * args.n_trials * 2
    if args.dry_run:
        print(json.dumps(
            {
                "input_path": str(args.input_path),
                "num_gold_items": len(gold_items),
                "providers": {
                    "openai": args.gpt_model,
                    "anthropic": args.claude_model,
                },
                "num_orderings": len(base.ORDERINGS),
                "num_eval_cases_per_provider": len(gold_items) * len(base.ORDERINGS) * args.n_trials,
                "planned_provider_calls": planned_calls,
                "min_score_margin": args.min_score_margin,
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
        ))
        return

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    openai_client = OpenAI()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.output_dir / f"gpt_claude_dpo_filter_{run_id}.jsonl"
    summary_path = args.output_dir / f"gpt_claude_dpo_filter_{run_id}_summary.json"

    experiment_config = {
        "run_id": run_id,
        "providers": {
            "openai": args.gpt_model,
            "anthropic": args.claude_model,
        },
        "input_path": str(args.input_path),
        "gold_labels": base.GOLD_LABELS,
        "n_trials": args.n_trials,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "claude_max_tokens": args.claude_max_tokens,
        "claude_version": args.claude_version,
        "min_score_margin": args.min_score_margin,
        "judge_prompt_type": "cross_provider_abstaining_few_shot_score_derived_dpo_filter",
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

    results: list[dict[str, Any]] = []
    for item in gold_items:
        for trial in range(1, args.n_trials + 1):
            for ordering in base.ORDERINGS:
                prompt = base.build_judge_prompt(item, ordering)

                providers = [
                    ("openai", args.gpt_model),
                    ("anthropic", args.claude_model),
                ]
                for provider, model in providers:
                    trial_start = time.time()
                    record_base = provider_record_base(
                        provider=provider,
                        model=model,
                        item=item,
                        ordering=ordering,
                        trial=trial,
                        experiment_config=experiment_config,
                    )
                    try:
                        if provider == "openai":
                            response = create_gpt_response(
                                openai_client,
                                model,
                                prompt,
                                args.temperature,
                                args.top_p,
                            )
                            row = parse_score_result(
                                base_record=record_base,
                                response_text=getattr(response, "output_text", None),
                                response_id=getattr(response, "id", None),
                                raw_response=None,
                                duration_sec=time.time() - trial_start,
                            )
                        else:
                            response = claude_judge.claude_message(
                                api_key=anthropic_api_key,
                                model=model,
                                prompt=prompt,
                                max_tokens=args.claude_max_tokens,
                                temperature=args.temperature,
                                anthropic_version=args.claude_version,
                            )
                            row = parse_score_result(
                                base_record=record_base,
                                response_text=claude_judge.output_text_from_claude_response(response),
                                response_id=response.get("id"),
                                raw_response=response,
                                duration_sec=time.time() - trial_start,
                            )
                        results.append(row)
                        base.append_jsonl(log_path, row)
                        print(
                            f"Saved {item['prompt_id']} | provider={provider} | model={model} "
                            f"| order={ordering['order_name']} | gold={item['gold_label']} "
                            f"| predicted={row['predicted_label']} | trial {trial}"
                        )
                    except Exception as exc:
                        row = {
                            **record_base,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "response_choice": base.INVALID_LABEL,
                            "predicted_label": base.INVALID_LABEL,
                            "correct": False,
                            "duration_sec": time.time() - trial_start,
                        }
                        results.append(row)
                        base.append_jsonl(log_path, row)
                        print(
                            f"{item['prompt_id']} | provider={provider} | model={model} "
                            f"| order={ordering['order_name']} | trial {trial} failed and was logged."
                        )

    metrics = base.compute_metrics(results)
    model_metrics = base.compute_model_metrics(results)
    order_bias_summary = base.compute_order_bias_summary(results)
    majority_vote_summary = base.compute_majority_vote_summary(results)
    provider_decisions = build_provider_decisions(results, args.min_score_margin)
    cross_provider_decisions = build_cross_provider_decisions(provider_decisions)
    cross_provider_summary = summarize_cross_provider_decisions(cross_provider_decisions)

    summary = {
        "config": experiment_config,
        "metrics": metrics,
        "model_metrics": model_metrics,
        "order_bias_summary": order_bias_summary,
        "majority_vote_summary": majority_vote_summary,
        "provider_stable_decisions": provider_decisions,
        "cross_provider_stable_dpo_summary": cross_provider_summary,
        "results": [
            {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "trial": row["trial"],
                "provider": row["provider"],
                "model": row["model"],
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
            "cross_provider_stable_dpo_summary": cross_provider_summary,
        },
    )

    base.print_run_summary(metrics, model_metrics, majority_vote_summary, order_bias_summary)
    print_cross_provider_summary(cross_provider_summary)
    print(f"Done. JSONL results saved to: {log_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
