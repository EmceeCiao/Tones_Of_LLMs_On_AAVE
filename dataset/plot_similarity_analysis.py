#!/usr/bin/env python3
"""
Simple plotting script for embedding_similarity_analysis.py outputs.

Usage:
    python3 plot_similarity_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("Final_Dataset")
DEFAULT_OUTPUT_DIR = Path("Final_Dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_pair_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No pair rows found in {path}")
    for row in rows:
        row["cosine_similarity"] = float(row["cosine_similarity"])
    return rows


def load_mismatch_means(path: Path) -> list[float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No mismatch rows found in {path}")
    return [float(row["mean_mismatch_cosine_similarity"]) for row in rows]


def make_plot(rows: list[dict], mismatch_means: list[float], output_path: Path) -> None:
    matched_scores = [row["cosine_similarity"] for row in rows]
    categories = sorted({row["category"] for row in rows})
    grouped = [[row["cosine_similarity"] for row in rows if row["category"] == category] for category in categories]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].hist(matched_scores, bins=min(10, max(5, len(matched_scores) // 2)), edgecolor="black")
    axes[0].set_title("Matched Pair Similarity")
    axes[0].set_xlabel("Cosine similarity")
    axes[0].set_ylabel("Count")

    axes[1].hist(mismatch_means, bins=20, edgecolor="black")
    axes[1].axvline(statistics.mean(matched_scores), color="red", linestyle="--", label="Matched mean")
    axes[1].set_title("Random-Mismatch Mean Distribution")
    axes[1].set_xlabel("Mean cosine similarity")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    axes[2].boxplot(grouped, tick_labels=categories)
    axes[2].set_title("Similarity by Category")
    axes[2].set_xlabel("Category")
    axes[2].set_ylabel("Cosine similarity")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[dict], mismatch_means: list[float], output_path: Path) -> None:
    matched_scores = [row["cosine_similarity"] for row in rows]
    matched_mean = statistics.mean(matched_scores)
    mismatch_mean = statistics.mean(mismatch_means)
    p_value = (sum(value >= matched_mean for value in mismatch_means) + 1) / (len(mismatch_means) + 1)

    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["cosine_similarity"])

    lines = [
        f"Pairs analyzed: {len(rows)}",
        f"Matched-pair mean cosine similarity: {matched_mean:.6f}",
        f"Matched-pair median cosine similarity: {statistics.median(matched_scores):.6f}",
        f"Random-mismatch mean cosine similarity: {mismatch_mean:.6f}",
        f"Mean difference (effect size): {matched_mean - mismatch_mean:.6f}",
        f"Permutation p-value (matched > mismatched): {p_value:.6f}",
        "",
        "By category:",
    ]

    for category in sorted(by_category):
        scores = by_category[category]
        lines.append(
            f"{category}: n={len(scores)}, mean={statistics.mean(scores):.6f}, median={statistics.median(scores):.6f}"
        )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    pair_csv = args.input_dir / "round1_embedding_similarity_results_v3_full.csv"
    mismatch_csv = args.input_dir / "round1_mismatch_distribution_v3_full.csv"
    output_image = args.output_dir / "round1_similarity_plots_v3_full.png"
    output_summary = args.output_dir / "round1_plot_summary_v3_full.txt"

    rows = load_pair_rows(pair_csv)
    mismatch_means = load_mismatch_means(mismatch_csv)
    make_plot(rows, mismatch_means, output_image)
    write_summary(rows, mismatch_means, output_summary)

    print(f"Plot written to: {output_image}")
    print(f"Summary written to: {output_summary}")


if __name__ == "__main__":
    main()
