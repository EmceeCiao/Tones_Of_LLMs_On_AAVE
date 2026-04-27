#!/usr/bin/env python3
"""
Analyze repeated SAE/AAVE runs using embeddings.

Input format:
    A JSON array of records like:
    {
      "prompt_index": 1,
      "prompt_text": "...",
      "dialect": "SAE" or "AAVE",
      "trial": 1,
      "response_text": "...",
      "reasoning_text": null
    }

What this script measures:
1. Within-dialect consistency:
   - How similar SAE trials are to other SAE trials for the same prompt
   - How similar AAVE trials are to other AAVE trials for the same prompt
2. Between-dialect matched-trial similarity:
   - SAE trial k vs AAVE trial k for the same prompt, when both exist
3. Between-dialect all-pairs similarity:
   - Every SAE trial vs every AAVE trial for the same prompt

Usage:
    export OPENAI_API_KEY="your-key"
    python3 analyze_repeated_runs_similarity.py --input filtered_responses.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from openai import OpenAI


DEFAULT_INPUT = Path("filtered_responses.json")
DEFAULT_OUTPUT_DIR = Path("study_logs_analysis")
DEFAULT_MODEL = "text-embedding-3-large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def get_embeddings(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in batched(texts, 64):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def load_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text())
    cleaned = []
    for item in records:
        response_text = (item.get("response_text") or "").strip()
        if not response_text:
            continue
        cleaned.append(
            {
                "prompt_index": item.get("prompt_index"),
                "prompt_text": item.get("prompt_text", ""),
                "category": item.get("category", ""),
                "dialect": item.get("dialect"),
                "trial": item.get("trial"),
                "response_text": response_text,
            }
        )
    return cleaned


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    prompt_rows: list[dict],
    overall: dict,
    category_rows: list[dict],
    model: str,
    input_path: Path,
) -> str:
    lines = [
        f"Input file: {input_path}",
        f"Embedding model: {model}",
        f"Prompts analyzed: {overall['num_prompts']}",
        f"Total response records analyzed: {overall['num_records']}",
        "",
        "Overall means:",
        f"Within-SAE trial similarity: {fmt(overall['within_sae_mean'])}",
        f"Within-AAVE trial similarity: {fmt(overall['within_aave_mean'])}",
        f"Between-dialect matched-trial similarity: {fmt(overall['between_matched_mean'])}",
        f"Between-dialect all-pairs similarity: {fmt(overall['between_all_pairs_mean'])}",
        "",
        "Interpretation:",
        "- Within-dialect scores estimate how much normal sampling variation exists across repeated runs.",
        "- Between-dialect scores estimate how similar SAE and AAVE outputs are for the same prompt.",
        "- If between-dialect similarity is close to within-dialect similarity, dialect differences are small relative to ordinary generation variability.",
        "- If between-dialect similarity is much lower than within-dialect similarity, dialect wording may be changing response content more substantially.",
        "",
        "By category:",
    ]

    for row in category_rows:
        lines.append(
            f"{row['category']}: prompts={row['num_prompts']}, "
            f"SAE-within={row['within_sae_mean']}, "
            f"AAVE-within={row['within_aave_mean']}, "
            f"matched={row['between_matched_mean']}, "
            f"all-pairs={row['between_all_pairs_mean']}"
        )

    lines.extend(
        [
            "",
        "Per-prompt means:",
        ]
    )

    for row in prompt_rows:
        lines.append(
            f"Prompt {row['prompt_index']}"
            f"{f' ({row['category']})' if row['category'] else ''}: SAE-within={row['within_sae_mean']}, "
            f"AAVE-within={row['within_aave_mean']}, matched={row['between_matched_mean']}, "
            f"all-pairs={row['between_all_pairs_mean']}"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    client = OpenAI()

    records = load_records(args.input)
    if not records:
        raise ValueError("No valid response records found.")

    texts = [record["response_text"] for record in records]
    vectors = get_embeddings(client, texts, args.model)
    for record, vector in zip(records, vectors):
        record["embedding"] = vector

    grouped: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["prompt_index"]][record["dialect"]].append(record)

    prompt_rows = []
    pairwise_rows = []

    for prompt_index in sorted(grouped):
        prompt_group = grouped[prompt_index]
        sae_runs = sorted(prompt_group.get("SAE", []), key=lambda x: x["trial"])
        aave_runs = sorted(prompt_group.get("AAVE", []), key=lambda x: x["trial"])

        within_sae = []
        within_aave = []
        between_matched = []
        between_all_pairs = []

        for left, right in itertools.combinations(sae_runs, 2):
            score = cosine_similarity(left["embedding"], right["embedding"])
            within_sae.append(score)
            pairwise_rows.append(
                {
                    "prompt_index": prompt_index,
                    "comparison_type": "within_sae",
                    "left_trial": left["trial"],
                    "right_trial": right["trial"],
                    "cosine_similarity": f"{score:.6f}",
                }
            )

        for left, right in itertools.combinations(aave_runs, 2):
            score = cosine_similarity(left["embedding"], right["embedding"])
            within_aave.append(score)
            pairwise_rows.append(
                {
                    "prompt_index": prompt_index,
                    "comparison_type": "within_aave",
                    "left_trial": left["trial"],
                    "right_trial": right["trial"],
                    "cosine_similarity": f"{score:.6f}",
                }
            )

        aave_by_trial = {row["trial"]: row for row in aave_runs}
        for sae_row in sae_runs:
            aave_row = aave_by_trial.get(sae_row["trial"])
            if aave_row is None:
                continue
            score = cosine_similarity(sae_row["embedding"], aave_row["embedding"])
            between_matched.append(score)
            pairwise_rows.append(
                {
                    "prompt_index": prompt_index,
                    "comparison_type": "between_matched",
                    "left_trial": sae_row["trial"],
                    "right_trial": aave_row["trial"],
                    "cosine_similarity": f"{score:.6f}",
                }
            )

        for sae_row in sae_runs:
            for aave_row in aave_runs:
                score = cosine_similarity(sae_row["embedding"], aave_row["embedding"])
                between_all_pairs.append(score)
                pairwise_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "comparison_type": "between_all_pairs",
                        "left_trial": sae_row["trial"],
                        "right_trial": aave_row["trial"],
                        "cosine_similarity": f"{score:.6f}",
                    }
                )

        prompt_rows.append(
            {
                "prompt_index": prompt_index,
                "category": sae_runs[0].get("category") or aave_runs[0].get("category") or "",
                "n_sae_trials": len(sae_runs),
                "n_aave_trials": len(aave_runs),
                "within_sae_mean": fmt(mean_or_none(within_sae)),
                "within_aave_mean": fmt(mean_or_none(within_aave)),
                "between_matched_mean": fmt(mean_or_none(between_matched)),
                "between_all_pairs_mean": fmt(mean_or_none(between_all_pairs)),
            }
        )

    overall_within_sae = [
        float(row["within_sae_mean"]) for row in prompt_rows if row["within_sae_mean"]
    ]
    overall_within_aave = [
        float(row["within_aave_mean"]) for row in prompt_rows if row["within_aave_mean"]
    ]
    overall_between_matched = [
        float(row["between_matched_mean"]) for row in prompt_rows if row["between_matched_mean"]
    ]
    overall_between_all_pairs = [
        float(row["between_all_pairs_mean"]) for row in prompt_rows if row["between_all_pairs_mean"]
    ]

    overall = {
        "num_prompts": len(prompt_rows),
        "num_records": len(records),
        "within_sae_mean": mean_or_none(overall_within_sae),
        "within_aave_mean": mean_or_none(overall_within_aave),
        "between_matched_mean": mean_or_none(overall_between_matched),
        "between_all_pairs_mean": mean_or_none(overall_between_all_pairs),
    }

    categories = sorted({row["category"] for row in prompt_rows if row["category"]})
    category_rows = []
    for category in categories:
        category_prompt_rows = [row for row in prompt_rows if row["category"] == category]
        category_rows.append(
            {
                "category": category,
                "num_prompts": len(category_prompt_rows),
                "within_sae_mean": fmt(
                    mean_or_none([float(row["within_sae_mean"]) for row in category_prompt_rows if row["within_sae_mean"]])
                ),
                "within_aave_mean": fmt(
                    mean_or_none([float(row["within_aave_mean"]) for row in category_prompt_rows if row["within_aave_mean"]])
                ),
                "between_matched_mean": fmt(
                    mean_or_none(
                        [float(row["between_matched_mean"]) for row in category_prompt_rows if row["between_matched_mean"]]
                    )
                ),
                "between_all_pairs_mean": fmt(
                    mean_or_none(
                        [float(row["between_all_pairs_mean"]) for row in category_prompt_rows if row["between_all_pairs_mean"]]
                    )
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_csv = args.output_dir / "repeated_runs_prompt_summary.csv"
    pairwise_csv = args.output_dir / "repeated_runs_pairwise_scores.csv"
    category_csv = args.output_dir / "repeated_runs_category_summary.csv"
    summary_txt = args.output_dir / "repeated_runs_summary.txt"

    write_csv(
        prompt_csv,
        prompt_rows,
        [
            "prompt_index",
            "category",
            "n_sae_trials",
            "n_aave_trials",
            "within_sae_mean",
            "within_aave_mean",
            "between_matched_mean",
            "between_all_pairs_mean",
        ],
    )
    write_csv(
        category_csv,
        category_rows,
        [
            "category",
            "num_prompts",
            "within_sae_mean",
            "within_aave_mean",
            "between_matched_mean",
            "between_all_pairs_mean",
        ],
    )
    write_csv(
        pairwise_csv,
        pairwise_rows,
        ["prompt_index", "comparison_type", "left_trial", "right_trial", "cosine_similarity"],
    )

    summary_text = build_summary(prompt_rows, overall, category_rows, args.model, args.input)
    summary_txt.write_text(summary_text + "\n")

    print(summary_text)
    print(f"\nPrompt summary written to: {prompt_csv}")
    print(f"Category summary written to: {category_csv}")
    print(f"Pairwise scores written to: {pairwise_csv}")
    print(f"Summary written to: {summary_txt}")


if __name__ == "__main__":
    main()
