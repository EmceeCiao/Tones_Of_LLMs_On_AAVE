#!/usr/bin/env python3
"""Evaluate a few-shot score-derived judge on the six clean Round 1 gold prompts.

The judge receives only the SAE/source prompt and the two response candidates
for each evaluated item. It does not receive the AAVE-translated prompt.

Variant 10 asks GPT-4.1 for rationale-informed dimension scores only. The script
derives the final label with a transparent deterministic rule instead of
trusting the model's direct winner.

This variant adds synthetic few-shot examples. They are not copied from any
Round 1 gold prompt or response.

This variant includes P05 so we can test whether synthetic few-shot calibration
helps with the detail/specificity vs readability/usability conflict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "Final_Dataset" / "final_prompt_responses.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "study_logs_judge"

DEFAULT_MODELS = ["gpt-4.1"]
TEMPERATURE = 0
TOP_P = 1
N_TRIALS = 1

LABELS = ["SAE", "AAVE", "No Preference"]
INVALID_LABEL = "Invalid"
TIE_LABEL = "Tie"
RESPONSE_CHOICES = ["Response A", "Response B", "No Preference"]

ORDERINGS = [
    {
        "order_name": "sae_first",
        "response_a_label": "SAE",
        "response_b_label": "AAVE",
    },
    {
        "order_name": "aave_first",
        "response_a_label": "AAVE",
        "response_b_label": "SAE",
    },
]

GOLD_LABELS = {
    "P02": "No Preference",
    "P03": "AAVE",
    "P05": "AAVE",
    "P08": "AAVE",
    "P09": "SAE",
    "P13": "No Preference",
}

EXCLUDED_GOLD_LABELS = {}

JUDGE_PROMPT = """You are scoring two assistant responses to the same user prompt.

Your task is to score each response on concrete dimensions. Do not choose the
final winner yourself; the script will derive the winner from your scores.

You are helping predict majority human preference from an evaluator study, not
rewarding the answer that is most formal, academic, or maximally detailed.

Human Round 1 rationales suggest these preference patterns:
- Raters often penalized responses that were technically richer but too dense,
  visually busy, overformatted, overwhelming, or too specialized for a general
  user question.
- Raters often valued organization, practical framing, ease of use, and a
  response that stayed focused on the user's actual question.
- Raters valued responses that covered the actual parts of the prompt. If a
  prompt asks about multiple items or concepts, a response that clearly addresses
  those parts can be preferable even if it is shorter.
- Raters penalized responses that over-assumed the user's situation, introduced
  unnecessary diagnosis, or reframed a general question into a more serious or
  narrower topic.
- Raters often chose No Preference when both responses answered the user
  similarly well, even if one had a small warmth, formatting, or detail
  advantage.

Consider:
- helpfulness
- clarity
- warmth
- appropriateness
- naturalness
- whether either response sounds forced, stereotyped, condescending, or socially inappropriate

Important:
- Do not treat formality, longer explanation, or more technical
  precision as automatically better. Those are advantages only when they improve
  the user's likely experience.
- Do not reward extra length or extra detail by itself. A shorter, more focused
  response can be better when it answers the prompt adequately and is easier for
  the user to use.
- Penalize answers that are hard to scan because of excessive bolding,
  overformatting, dense detail, or unclear focus.
- Do not penalize sectioning, bullets, or separate subsections when they make the
  answer easier to scan or help distinguish different parts of the user's
  question.
- Factual correctness and task completion matter most when one response is
  clearly wrong or incomplete. If both responses answer the question adequately,
  give real weight to accessibility, naturalness, and how well the response fits
  the user's likely needs.
- Do not penalize dialectal language simply for being dialectal.
- Do penalize language that sounds forced, exaggerated, stereotyped,
  condescending, or unnatural.
- Treat over-assumption, unnecessary diagnosis, or moving the answer away from
  the user's actual question as a serious flaw.
- Use the same score when differences are small or unlikely to matter to a human
  evaluator. Do not manufacture score differences for tiny style variations.
- A one-point difference should mean a noticeable practical difference.
- Be internally consistent when scoring: a response should not become better or
  worse just because it appears as Response A instead of Response B.

Scoring scale:
1 = poor
2 = weak
3 = acceptable
4 = good
5 = excellent

Few-shot calibration examples:

Example 1: choose effectively equal scores when both answers are similarly useful.

User prompt:
How do I reset my account password?

Response A:
Go to Settings, choose Account, select Password, and follow the reset steps. If
you cannot log in, use the "Forgot password" link on the sign-in page.

Response B:
Open your account settings and pick Password. If you're locked out, use "Forgot
password" from the login screen and follow the email instructions.

Expected JSON:
{{
  "response_a_scores": {{
    "task_completion_correctness": 5,
    "clarity": 5,
    "task_coverage": 5,
    "warmth": 3,
    "accessibility": 5,
    "usable_detail_level": 5,
    "naturalness_appropriateness": 5,
    "fit_to_user_prompt": 5
  }},
  "response_b_scores": {{
    "task_completion_correctness": 5,
    "clarity": 5,
    "task_coverage": 5,
    "warmth": 3,
    "accessibility": 5,
    "usable_detail_level": 5,
    "naturalness_appropriateness": 5,
    "fit_to_user_prompt": 5
  }},
  "serious_flaw": "None",
  "concerns": [],
  "score_rationale": "Both responses give the same practical steps clearly. Small wording differences should not create a preference."
}}

Example 2: reward usable detail and task coverage, not extra length by itself.

User prompt:
What happens when I delete a photo and an app from my phone?

Response A:
When you delete something, the operating system changes filesystem references,
may retain blocks until overwritten, can interact with flash storage garbage
collection, and may preserve copies in caches, backups, or cloud sync layers.
The exact behavior depends on the filesystem, device, and deletion API.

Response B:
For a photo, it usually moves to a Recently Deleted or Trash folder first, then
gets removed later. For an app, the app is uninstalled, but some settings or
cloud data may remain. In both cases, deletion often means the device stops
showing it before every copy is truly gone.

Expected JSON:
{{
  "response_a_scores": {{
    "task_completion_correctness": 4,
    "clarity": 3,
    "task_coverage": 3,
    "warmth": 2,
    "accessibility": 2,
    "usable_detail_level": 2,
    "naturalness_appropriateness": 4,
    "fit_to_user_prompt": 3
  }},
  "response_b_scores": {{
    "task_completion_correctness": 5,
    "clarity": 5,
    "task_coverage": 5,
    "warmth": 3,
    "accessibility": 5,
    "usable_detail_level": 5,
    "naturalness_appropriateness": 5,
    "fit_to_user_prompt": 5
  }},
  "serious_flaw": "None",
  "concerns": [],
  "score_rationale": "Response B better covers both photos and apps in an accessible way. Response A is technical but less usable for the likely user."
}}

Example 3: penalize over-assumption or unnecessary diagnosis.

User prompt:
Why do people feel tired in the morning but more awake at night?

Response A:
That can happen because of circadian rhythm, light exposure, inconsistent sleep,
caffeine timing, stress, or habits that push alertness later into the day.

Response B:
This is common for people with depression or anxiety. Your brain may be avoiding
the day, and you might need to talk to a mental health professional.

Expected JSON:
{{
  "response_a_scores": {{
    "task_completion_correctness": 5,
    "clarity": 5,
    "task_coverage": 5,
    "warmth": 3,
    "accessibility": 5,
    "usable_detail_level": 5,
    "naturalness_appropriateness": 5,
    "fit_to_user_prompt": 5
  }},
  "response_b_scores": {{
    "task_completion_correctness": 2,
    "clarity": 3,
    "task_coverage": 2,
    "warmth": 3,
    "accessibility": 3,
    "usable_detail_level": 2,
    "naturalness_appropriateness": 2,
    "fit_to_user_prompt": 2
  }},
  "serious_flaw": "Response B",
  "concerns": ["over-assumption", "unnecessary diagnosis", "too narrow"],
  "score_rationale": "Response B reframes a general question as a mental health issue without enough basis. Response A answers the broad question more appropriately."
}}

Now score the actual responses. Return valid JSON only, with no Markdown:

{{
  "response_a_scores": {{
    "task_completion_correctness": 1,
    "clarity": 1,
    "task_coverage": 1,
    "warmth": 1,
    "accessibility": 1,
    "usable_detail_level": 1,
    "naturalness_appropriateness": 1,
    "fit_to_user_prompt": 1
  }},
  "response_b_scores": {{
    "task_completion_correctness": 1,
    "clarity": 1,
    "task_coverage": 1,
    "warmth": 1,
    "accessibility": 1,
    "usable_detail_level": 1,
    "naturalness_appropriateness": 1,
    "fit_to_user_prompt": 1
  }},
  "serious_flaw": "Response A" | "Response B" | "Both" | "None",
  "concerns": [],
  "score_rationale": "Brief explanation of the score differences."
}}

User prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}
"""


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_gold_items(input_path: Path) -> list[dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "round 1" not in data:
        raise ValueError('Expected input JSON to contain a "round 1" list.')

    items = []
    for index, item in enumerate(data["round 1"], start=1):
        prompt_id = f"P{index:02d}"
        if prompt_id not in GOLD_LABELS:
            continue

        missing_fields = [
            field
            for field in ("sae_prompt", "sae_response", "aave_response")
            if not item.get(field)
        ]
        if missing_fields:
            raise ValueError(f"{prompt_id} is missing required fields: {missing_fields}")

        items.append(
            {
                "prompt_id": prompt_id,
                "category": item.get("category", ""),
                "sae_prompt": item["sae_prompt"],
                "sae_response": item["sae_response"],
                "aave_response": item["aave_response"],
                "gold_label": GOLD_LABELS[prompt_id],
            }
        )

    found_ids = {item["prompt_id"] for item in items}
    missing_ids = sorted(set(GOLD_LABELS) - found_ids)
    if missing_ids:
        raise ValueError(f"Could not find gold prompt IDs in input: {missing_ids}")

    return sorted(items, key=lambda item: item["prompt_id"])


def response_text_for_label(item: dict[str, Any], label: str) -> str:
    if label == "SAE":
        return item["sae_response"]
    if label == "AAVE":
        return item["aave_response"]
    raise ValueError(f"Unsupported response label: {label}")


def build_judge_prompt(item: dict[str, Any], ordering: dict[str, str]) -> str:
    return JUDGE_PROMPT.format(
        prompt=item["sae_prompt"],
        response_a=response_text_for_label(item, ordering["response_a_label"]),
        response_b=response_text_for_label(item, ordering["response_b_label"]),
    )


def strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def parse_judge_output(text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not text:
        return None, "empty response"

    try:
        parsed = json.loads(strip_json_fence(text))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"

    if not isinstance(parsed, dict):
        return None, "JSON output is not an object"

    return parsed, None


def normalize_response_choice(value: Any) -> str:
    if not isinstance(value, str):
        return INVALID_LABEL

    normalized = re.sub(r"\s+", " ", value.strip())
    choice_lookup = {
        "response a": "Response A",
        "a": "Response A",
        "response b": "Response B",
        "b": "Response B",
        "no preference": "No Preference",
    }
    return choice_lookup.get(normalized.casefold(), INVALID_LABEL)


def response_choice_to_label(response_choice: str, ordering: dict[str, str]) -> str:
    if response_choice == "Response A":
        return ordering["response_a_label"]
    if response_choice == "Response B":
        return ordering["response_b_label"]
    if response_choice == "No Preference":
        return "No Preference"
    return INVALID_LABEL


SCORE_FIELDS = [
    "task_completion_correctness",
    "clarity",
    "task_coverage",
    "warmth",
    "accessibility",
    "usable_detail_level",
    "naturalness_appropriateness",
    "fit_to_user_prompt",
]


def numeric_score(value: Any) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 5:
        return score
    return None


def extract_scores(judge_result: dict[str, Any] | None, key: str) -> dict[str, int | None]:
    raw_scores = judge_result.get(key, {}) if judge_result else {}
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    return {
        field: numeric_score(raw_scores.get(field))
        for field in SCORE_FIELDS
    }


def safe_total(scores: dict[str, int | None]) -> int | None:
    values = list(scores.values())
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def dimension_leads(
    response_a_scores: dict[str, int | None],
    response_b_scores: dict[str, int | None],
) -> dict[str, str]:
    leads = {}
    for field in SCORE_FIELDS:
        a_score = response_a_scores[field]
        b_score = response_b_scores[field]
        if a_score is None or b_score is None or a_score == b_score:
            leads[field] = "Tie"
        elif a_score > b_score:
            leads[field] = "Response A"
        else:
            leads[field] = "Response B"
    return leads


def derive_response_choice_from_scores(
    judge_result: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    response_a_scores = extract_scores(judge_result, "response_a_scores")
    response_b_scores = extract_scores(judge_result, "response_b_scores")
    response_a_total = safe_total(response_a_scores)
    response_b_total = safe_total(response_b_scores)
    serious_flaw = judge_result.get("serious_flaw") if judge_result else None
    serious_flaw = serious_flaw if serious_flaw in {"Response A", "Response B", "Both", "None"} else "None"
    leads = dimension_leads(response_a_scores, response_b_scores)

    derivation = {
        "response_a_scores": response_a_scores,
        "response_b_scores": response_b_scores,
        "response_a_total": response_a_total,
        "response_b_total": response_b_total,
        "dimension_leads": leads,
        "serious_flaw": serious_flaw,
        "rule": "serious flaw, else total gap >=2, else 1-point gap with fit plus clarity/task coverage/accessibility/usable detail, else No Preference",
    }

    if response_a_total is None or response_b_total is None:
        return INVALID_LABEL, derivation

    if serious_flaw == "Response A":
        return "Response B", derivation
    if serious_flaw == "Response B":
        return "Response A", derivation
    if serious_flaw == "Both":
        return "No Preference", derivation

    diff = response_a_total - response_b_total
    abs_diff = abs(diff)
    if abs_diff >= 2:
        return ("Response A" if diff > 0 else "Response B"), derivation

    if abs_diff == 1:
        leader = "Response A" if diff > 0 else "Response B"
        meaningful_fields = {
            "task_completion_correctness",
            "clarity",
            "task_coverage",
            "accessibility",
            "usable_detail_level",
            "fit_to_user_prompt",
        }
        leader_meaningful_edges = sum(
            1 for field in meaningful_fields if leads[field] == leader
        )
        if leads["fit_to_user_prompt"] == leader and leader_meaningful_edges >= 2:
            return leader, derivation

    return "No Preference", derivation


def collect_reasoning_text(response: Any) -> str | None:
    reasoning_texts = []
    for output_item in getattr(response, "output", []) or []:
        if getattr(output_item, "type", None) == "reasoning":
            for summary in getattr(output_item, "summary", []) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning_texts.append(text)
    return "\n".join(reasoning_texts) if reasoning_texts else None


def compute_confusion_matrix(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    predicted_labels = LABELS + [INVALID_LABEL, TIE_LABEL]
    matrix = {
        gold_label: {predicted_label: 0 for predicted_label in predicted_labels}
        for gold_label in LABELS
    }

    for row in results:
        gold_label = row["gold_label"]
        predicted_label = row["predicted_label"]
        if predicted_label not in predicted_labels:
            predicted_label = INVALID_LABEL
        matrix[gold_label][predicted_label] += 1

    return matrix


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for label in LABELS:
        true_positive = sum(
            1
            for row in results
            if row["gold_label"] == label and row["predicted_label"] == label
        )
        false_positive = sum(
            1
            for row in results
            if row["gold_label"] != label and row["predicted_label"] == label
        )
        false_negative = sum(
            1
            for row in results
            if row["gold_label"] == label and row["predicted_label"] != label
        )
        support = sum(1 for row in results if row["gold_label"] == label)
        predicted_count = sum(1 for row in results if row["predicted_label"] == label)

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )

        metrics[label] = {
            "support": support,
            "predicted_count": predicted_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    correct = sum(1 for row in results if row["gold_label"] == row["predicted_label"])
    metrics["overall"] = {
        "total": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else None,
        "invalid_predictions": sum(1 for row in results if row["predicted_label"] == INVALID_LABEL),
        "tie_predictions": sum(1 for row in results if row["predicted_label"] == TIE_LABEL),
    }
    metrics["confusion_matrix"] = compute_confusion_matrix(results)
    return metrics


def compute_order_bias_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_order = {}
    for order_name in sorted({row["order_name"] for row in results}):
        order_results = [row for row in results if row["order_name"] == order_name]
        by_order[order_name] = compute_metrics(order_results)

    by_prompt_trial_model: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in results:
        by_prompt_trial_model.setdefault((row["prompt_id"], row["trial"], row["model"]), []).append(row)

    consistency_rows = []
    for (prompt_id, trial, model), rows in sorted(by_prompt_trial_model.items()):
        predictions = {
            row["order_name"]: row["predicted_label"]
            for row in rows
        }
        response_choices = {
            row["order_name"]: row["response_choice"]
            for row in rows
        }
        consistency_rows.append(
            {
                "prompt_id": prompt_id,
                "trial": trial,
                "model": model,
                "gold_label": rows[0]["gold_label"],
                "predictions_by_order": predictions,
                "response_choices_by_order": response_choices,
                "consistent_prediction": len(set(predictions.values())) == 1,
            }
        )

    return {
        "metrics_by_order": by_order,
        "prompt_order_consistency": consistency_rows,
        "consistent_prompt_trials": sum(
            1 for row in consistency_rows if row["consistent_prediction"]
        ),
        "total_prompt_trials": len(consistency_rows),
    }


def majority_label(votes: list[str]) -> str:
    valid_votes = [vote for vote in votes if vote in LABELS]
    if not valid_votes:
        return INVALID_LABEL

    counts = {label: valid_votes.count(label) for label in LABELS}
    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    return winners[0] if len(winners) == 1 else TIE_LABEL


def compute_model_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    for model in sorted({row["model"] for row in results}):
        model_results = [row for row in results if row["model"] == model]
        by_order = {}
        for order_name in sorted({row["order_name"] for row in model_results}):
            by_order[order_name] = compute_metrics(
                [row for row in model_results if row["order_name"] == order_name]
            )
        by_model[model] = {
            "overall": compute_metrics(model_results),
            "by_order": by_order,
        }
    return by_model


def build_majority_vote_rows(
    results: list[dict[str, Any]], group_fields: tuple[str, ...], vote_scope: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in results:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    vote_rows = []
    for key, rows in sorted(grouped.items()):
        field_values = dict(zip(group_fields, key))
        votes = [row["predicted_label"] for row in rows]
        predicted_label = majority_label(votes)
        first_row = rows[0]
        vote_rows.append(
            {
                **field_values,
                "vote_scope": vote_scope,
                "prompt_id": first_row["prompt_id"],
                "category": first_row["category"],
                "trial": first_row["trial"],
                "gold_label": first_row["gold_label"],
                "predicted_label": predicted_label,
                "correct": predicted_label == first_row["gold_label"],
                "votes": votes,
                "vote_counts": {label: votes.count(label) for label in LABELS + [INVALID_LABEL]},
                "num_votes": len(votes),
                "models": sorted({row["model"] for row in rows}),
            }
        )
    return vote_rows


def compute_majority_vote_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_order_rows = build_majority_vote_rows(
        results,
        ("prompt_id", "trial", "order_name"),
        "models_by_prompt_trial_order",
    )
    across_order_rows = build_majority_vote_rows(
        results,
        ("prompt_id", "trial"),
        "models_and_orders_by_prompt_trial",
    )

    consistency_rows = []
    by_prompt_trial: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in by_order_rows:
        by_prompt_trial.setdefault((row["prompt_id"], row["trial"]), []).append(row)

    for (prompt_id, trial), rows in sorted(by_prompt_trial.items()):
        predictions = {row["order_name"]: row["predicted_label"] for row in rows}
        consistency_rows.append(
            {
                "prompt_id": prompt_id,
                "trial": trial,
                "gold_label": rows[0]["gold_label"],
                "majority_predictions_by_order": predictions,
                "consistent_majority_across_orders": len(set(predictions.values())) == 1,
            }
        )

    return {
        "by_order_results": by_order_rows,
        "by_order_metrics": compute_metrics(by_order_rows),
        "across_order_results": across_order_rows,
        "across_order_metrics": compute_metrics(across_order_rows),
        "order_consistency": consistency_rows,
        "consistent_majority_prompt_trials": sum(
            1 for row in consistency_rows if row["consistent_majority_across_orders"]
        ),
        "total_prompt_trials": len(consistency_rows),
    }


def fmt_rate(value: Any) -> str:
    return "NA" if value is None else f"{value:.3f}"


def print_metrics_block(title: str, metrics: dict[str, Any]) -> None:
    overall = metrics["overall"]
    print(f"\n{title}")
    print(
        f"  accuracy: {overall['correct']}/{overall['total']} "
        f"({fmt_rate(overall['accuracy'])})"
    )
    if overall.get("invalid_predictions") or overall.get("tie_predictions"):
        print(
            f"  invalid: {overall.get('invalid_predictions', 0)} "
            f"| ties: {overall.get('tie_predictions', 0)}"
        )
    for label in LABELS:
        row = metrics[label]
        print(
            f"  {label}: support={row['support']} predicted={row['predicted_count']} "
            f"precision={fmt_rate(row['precision'])} recall={fmt_rate(row['recall'])} "
            f"f1={fmt_rate(row['f1'])}"
        )
    print("  confusion matrix:")
    for gold_label, predictions in metrics["confusion_matrix"].items():
        counts = ", ".join(
            f"{predicted_label}={count}"
            for predicted_label, count in predictions.items()
            if count
        )
        print(f"    gold {gold_label}: {counts or 'all zero'}")


def print_run_summary(
    metrics: dict[str, Any],
    model_metrics: dict[str, Any],
    majority_vote_summary: dict[str, Any],
    order_bias_summary: dict[str, Any],
) -> None:
    print("\n=== Judge Run Summary ===")
    print_metrics_block("Individual judge calls, pooled", metrics)

    for model, model_summary in model_metrics.items():
        print_metrics_block(f"Model {model}, pooled orders", model_summary["overall"])

    print_metrics_block(
        "Majority vote across models, per response order",
        majority_vote_summary["by_order_metrics"],
    )
    print_metrics_block(
        "Majority vote across models and both response orders",
        majority_vote_summary["across_order_metrics"],
    )

    print("\nOrder consistency")
    print(
        "  individual calls consistent by prompt/trial: "
        f"{order_bias_summary['consistent_prompt_trials']}/"
        f"{order_bias_summary['total_prompt_trials']}"
    )
    print(
        "  majority vote consistent by prompt/trial: "
        f"{majority_vote_summary['consistent_majority_prompt_trials']}/"
        f"{majority_vote_summary['total_prompt_trials']}"
    )
    for row in majority_vote_summary["order_consistency"]:
        if not row["consistent_majority_across_orders"]:
            print(
                f"  order-sensitive majority: {row['prompt_id']} "
                f"gold={row['gold_label']} "
                f"predictions={row['majority_predictions_by_order']}"
            )


def create_response(client: Any, model: str, prompt: str, temperature: float, top_p: float) -> Any:
    if model.startswith("gpt-5"):
        return client.responses.create(
            model=model,
            input=prompt,
        )

    return client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a zero-shot single LLM-as-judge test on the six clean Round 1 gold prompts."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Judge models to run. Default: gpt-4.1.",
    )
    parser.add_argument(
        "--model",
        dest="single_model",
        default=None,
        help="Run one judge model. Overrides --models.",
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the six gold prompts and print their IDs without calling OpenAI.",
    )
    args = parser.parse_args()
    if args.single_model:
        args.models = [args.single_model]
    return args


def main() -> None:
    args = parse_args()
    gold_items = load_gold_items(args.input_path)

    if args.dry_run:
        print(json.dumps(
            {
                "input_path": str(args.input_path),
                "num_gold_items": len(gold_items),
                "excluded_gold_labels": EXCLUDED_GOLD_LABELS,
                "models": args.models,
                "num_models": len(args.models),
                "num_orderings": len(ORDERINGS),
                "num_eval_cases": len(gold_items) * len(ORDERINGS) * len(args.models) * args.n_trials,
                "orderings": ORDERINGS,
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

    from openai import OpenAI

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.output_dir / f"single_llm_judge10_{run_id}.jsonl"
    summary_path = args.output_dir / f"single_llm_judge10_{run_id}_summary.json"

    experiment_config = {
        "run_id": run_id,
        "models": args.models,
        "input_path": str(args.input_path),
        "gold_labels": GOLD_LABELS,
        "excluded_gold_labels": EXCLUDED_GOLD_LABELS,
        "num_gold_items": len(gold_items),
        "n_trials": args.n_trials,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "judge_prompt_type": "v10_few_shot_clean_gold_without_p05_score_derived_label_gpt41_default",
        "score_fields": SCORE_FIELDS,
        "orderings": ORDERINGS,
    }

    append_jsonl(
        log_path,
        {
            "record_type": "experiment_config",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": experiment_config,
        },
    )

    client = OpenAI()
    results = []

    for item in gold_items:
        for trial in range(1, args.n_trials + 1):
            for model in args.models:
                for ordering in ORDERINGS:
                    trial_start = time.time()
                    prompt = build_judge_prompt(item, ordering)
                    order_name = ordering["order_name"]
                    base_record = {
                        "record_type": "trial_result",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "config": experiment_config,
                        "prompt_id": item["prompt_id"],
                        "category": item["category"],
                        "gold_label": item["gold_label"],
                        "trial": trial,
                        "model": model,
                        "order_name": order_name,
                        "response_a_label": ordering["response_a_label"],
                        "response_b_label": ordering["response_b_label"],
                    }

                    try:
                        response = create_response(client, model, prompt, args.temperature, args.top_p)
                        response_text = getattr(response, "output_text", None)
                        judge_result, parse_error = parse_judge_output(response_text)
                        response_choice, derivation = derive_response_choice_from_scores(judge_result)
                        predicted_label = response_choice_to_label(response_choice, ordering)
                        result_row = {
                            **base_record,
                            "status": "success" if parse_error is None else "parse_error",
                            "response_id": getattr(response, "id", None),
                            "response_text": response_text,
                            "system_fingerprint": getattr(response, "system_fingerprint", None),
                            "reasoning_text": collect_reasoning_text(response),
                            "parse_error": parse_error,
                            "judge_result": judge_result,
                            "score_derivation": derivation,
                            "response_choice": response_choice,
                            "predicted_label": predicted_label,
                            "confidence": judge_result.get("confidence") if judge_result else None,
                            "correct": predicted_label == item["gold_label"],
                            "duration_sec": time.time() - trial_start,
                        }
                        results.append(result_row)
                        append_jsonl(log_path, result_row)
                        print(
                            f"Saved {item['prompt_id']} | model={model} | order={order_name} "
                            f"| gold={item['gold_label']} | predicted={predicted_label} "
                            f"| trial {trial}"
                        )

                    except Exception as exc:
                        result_row = {
                            **base_record,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "response_choice": INVALID_LABEL,
                            "predicted_label": INVALID_LABEL,
                            "correct": False,
                            "duration_sec": time.time() - trial_start,
                        }
                        results.append(result_row)
                        append_jsonl(log_path, result_row)
                        print(f"{item['prompt_id']} | model={model} | order={order_name} | trial {trial} failed and was logged.")

    metrics = compute_metrics(results)
    model_metrics = compute_model_metrics(results)
    order_bias_summary = compute_order_bias_summary(results)
    majority_vote_summary = compute_majority_vote_summary(results)
    summary = {
        "config": experiment_config,
        "metrics": metrics,
        "model_metrics": model_metrics,
        "order_bias_summary": order_bias_summary,
        "majority_vote_summary": majority_vote_summary,
        "results": [
            {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "trial": row["trial"],
                "model": row["model"],
                "order_name": row["order_name"],
                "response_a_label": row["response_a_label"],
                "response_b_label": row["response_b_label"],
                "gold_label": row["gold_label"],
                "response_choice": row.get("response_choice"),
                "predicted_label": row["predicted_label"],
                "confidence": row.get("confidence"),
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

    append_jsonl(
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
        },
    )

    print_run_summary(metrics, model_metrics, majority_vote_summary, order_bias_summary)
    print(f"Done. JSONL results saved to: {log_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
