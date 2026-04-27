#!/usr/bin/env python3
"""
Simple plotting companion for analyze_repeated_runs_similarity.py.

Reads:
    study_logs_analysis/repeated_runs_prompt_summary.csv
    study_logs_analysis/repeated_runs_pairwise_scores.csv
    study_logs_analysis/repeated_runs_category_summary.csv

Writes:
    study_logs_analysis/repeated_runs_similarity_plots.png
    study_logs_analysis/repeated_runs_plot_summary.txt

Usage:
    python3 plot_repeated_runs_similarity.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("study_logs_analysis")
DEFAULT_OUTPUT_DIR = Path("study_logs_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_prompt_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prompt rows found in {path}")
    return rows


def load_pairwise_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No pairwise rows found in {path}")
    for row in rows:
        row["prompt_index"] = int(row["prompt_index"])
        row["left_trial"] = int(row["left_trial"])
        row["right_trial"] = int(row["right_trial"])
        row["cosine_similarity"] = float(row["cosine_similarity"])
    return rows


def load_category_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No category rows found in {path}")
    return rows


def values_for(rows: list[dict], comparison_type: str) -> list[float]:
    return [row["cosine_similarity"] for row in rows if row["comparison_type"] == comparison_type]


def make_plot(
    pairwise_rows: list[dict],
    prompt_rows: list[dict],
    category_rows: list[dict],
    output_path: Path,
) -> None:
    within_sae = values_for(pairwise_rows, "within_sae")
    within_aave = values_for(pairwise_rows, "within_aave")
    between_matched = values_for(pairwise_rows, "between_matched")
    between_all_pairs = values_for(pairwise_rows, "between_all_pairs")

    prompt_indices = [int(row["prompt_index"]) for row in prompt_rows]
    prompt_within_sae = [float(row["within_sae_mean"]) for row in prompt_rows if row["within_sae_mean"]]
    prompt_within_aave = [float(row["within_aave_mean"]) for row in prompt_rows if row["within_aave_mean"]]
    prompt_between_matched = [float(row["between_matched_mean"]) for row in prompt_rows if row["between_matched_mean"]]
    prompt_between_all_pairs = [float(row["between_all_pairs_mean"]) for row in prompt_rows if row["between_all_pairs_mean"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    boxplot_data = [within_sae, within_aave, between_matched, between_all_pairs]
    boxplot_labels = ["Within SAE", "Within AAVE", "Between matched", "Between all pairs"]
    axes[0].boxplot(boxplot_data, tick_labels=boxplot_labels)
    axes[0].set_title("Pairwise Similarity Distributions")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].tick_params(axis="x", rotation=15)

    bins = 20
    axes[1].hist(within_sae, bins=bins, alpha=0.5, label="Within SAE", edgecolor="black")
    axes[1].hist(within_aave, bins=bins, alpha=0.5, label="Within AAVE", edgecolor="black")
    axes[1].hist(between_matched, bins=bins, alpha=0.5, label="Between matched", edgecolor="black")
    axes[1].set_title("Similarity Histogram")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    axes[2].plot(prompt_indices[: len(prompt_within_sae)], prompt_within_sae, marker="o", label="Within SAE")
    axes[2].plot(prompt_indices[: len(prompt_within_aave)], prompt_within_aave, marker="o", label="Within AAVE")
    axes[2].plot(
        prompt_indices[: len(prompt_between_matched)],
        prompt_between_matched,
        marker="o",
        label="Between matched",
    )
    axes[2].plot(
        prompt_indices[: len(prompt_between_all_pairs)],
        prompt_between_all_pairs,
        marker="o",
        label="Between all pairs",
    )
    axes[2].set_title("Per-Prompt Mean Similarity")
    axes[2].set_xlabel("Prompt index")
    axes[2].set_ylabel("Mean cosine similarity")
    axes[2].legend()

    categories = [row["category"] for row in category_rows]
    cat_within_sae = [float(row["within_sae_mean"]) for row in category_rows if row["within_sae_mean"]]
    cat_within_aave = [float(row["within_aave_mean"]) for row in category_rows if row["within_aave_mean"]]
    cat_between_matched = [float(row["between_matched_mean"]) for row in category_rows if row["between_matched_mean"]]

    x = list(range(len(categories)))
    width = 0.25
    axes[3].bar([i - width for i in x], cat_within_sae, width=width, label="Within SAE")
    axes[3].bar(x, cat_within_aave, width=width, label="Within AAVE")
    axes[3].bar([i + width for i in x], cat_between_matched, width=width, label="Between matched")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(categories)
    axes[3].set_title("Category-Level Mean Similarity")
    axes[3].set_ylabel("Mean cosine similarity")
    axes[3].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(pairwise_rows: list[dict], category_rows: list[dict], output_path: Path) -> None:
    within_sae = values_for(pairwise_rows, "within_sae")
    within_aave = values_for(pairwise_rows, "within_aave")
    between_matched = values_for(pairwise_rows, "between_matched")
    between_all_pairs = values_for(pairwise_rows, "between_all_pairs")

    lines = [
        f"Within-SAE mean similarity: {statistics.mean(within_sae):.6f}" if within_sae else "Within-SAE mean similarity: ",
        f"Within-AAVE mean similarity: {statistics.mean(within_aave):.6f}" if within_aave else "Within-AAVE mean similarity: ",
        f"Between-matched mean similarity: {statistics.mean(between_matched):.6f}" if between_matched else "Between-matched mean similarity: ",
        f"Between-all-pairs mean similarity: {statistics.mean(between_all_pairs):.6f}" if between_all_pairs else "Between-all-pairs mean similarity: ",
        "",
        "By category:",
    ]

    for row in category_rows:
        lines.append(
            f"{row['category']}: SAE-within={row['within_sae_mean']}, "
            f"AAVE-within={row['within_aave_mean']}, "
            f"between-matched={row['between_matched_mean']}, "
            f"between-all-pairs={row['between_all_pairs_mean']}"
        )

    lines.extend([
        "",
        "Reading guide:",
        "- Compare within-SAE and within-AAVE against between-matched.",
        "- If between-matched is close to the within-dialect values, dialect effects are small relative to normal repeated-run variation.",
        "- If between-matched is much lower, dialect condition is changing output semantics more than ordinary sampling noise.",
    ])

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    prompt_csv = args.input_dir / "repeated_runs_prompt_summary.csv"
    pairwise_csv = args.input_dir / "repeated_runs_pairwise_scores.csv"
    category_csv = args.input_dir / "repeated_runs_category_summary.csv"
    output_image = args.output_dir / "repeated_runs_similarity_plots.png"
    output_summary = args.output_dir / "repeated_runs_plot_summary.txt"

    prompt_rows = load_prompt_rows(prompt_csv)
    pairwise_rows = load_pairwise_rows(pairwise_csv)
    category_rows = load_category_rows(category_csv)
    make_plot(pairwise_rows, prompt_rows, category_rows, output_image)
    write_summary(pairwise_rows, category_rows, output_summary)

    print(f"Plot written to: {output_image}")
    print(f"Summary written to: {output_summary}")


if __name__ == "__main__":
    main()
