#!/usr/bin/env python3
"""
Raw full-response embedding similarity analysis

This script:
- reads the SAE/AAVE responses from Final_Dataset/final_prompt_responses.json
- embeds the raw full responses
- computes cosine similarity for each matched pair
- builds a random-mismatch baseline using a permutation test
- writes one per-pair CSV, one mismatch CSV, and one text summary

Usage:
    export OPENAI_API_KEY="your-key"
    python3 embedding_similarity_round1_v3.py

Optional:
    python3 embedding_similarity_round1_v3.py --model text-embedding-3-large --permutations 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from openai import OpenAI


DATA_PATH = Path("Final_Dataset/final_prompt_responses.json")
DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_PERMUTATIONS = 5000
DEFAULT_OUTPUT_DIR = Path("Final_Dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_round_1_pairs(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    pairs = []
    for index, item in enumerate(data["round 1"], start=1):
        sae = item["sae_response"].strip()
        aave = item["aave_response"].strip()
        if not sae or not aave:
            continue
        pairs.append(
            {
                "pair_id": index,
                "category": item.get("category", ""),
                "sae_response": sae,
                "aave_response": aave,
            }
        )
    return pairs


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def get_embeddings(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in batched(texts, 64):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(row.embedding for row in response.data)
    return vectors


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def derangement(n: int, rng: random.Random) -> list[int]:
    if n < 2:
        raise ValueError("Need at least two pairs for the permutation baseline.")
    while True:
        order = list(range(n))
        rng.shuffle(order)
        if all(i != order[i] for i in range(n)):
            return order


def mismatch_distribution(
    sae_vectors: list[list[float]],
    aave_vectors: list[list[float]],
    permutations: int,
    rng: random.Random,
) -> list[float]:
    means = []
    for _ in range(permutations):
        shuffled = derangement(len(aave_vectors), rng)
        scores = [
            cosine_similarity(sae_vectors[i], aave_vectors[shuffled[i]])
            for i in range(len(sae_vectors))
        ]
        means.append(statistics.mean(scores))
    return means


def fmt(value: float) -> str:
    return f"{value:.6f}"


def write_pair_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "category",
                "cosine_similarity",
                "sae_response_chars",
                "aave_response_chars",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_mismatch_csv(path: Path, mismatch_means: list[float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["permutation_id", "mean_mismatch_cosine_similarity"])
        writer.writeheader()
        for index, value in enumerate(mismatch_means, start=1):
            writer.writerow(
                {
                    "permutation_id": index,
                    "mean_mismatch_cosine_similarity": fmt(value),
                }
            )


def build_summary(rows: list[dict], mismatch_means: list[float], model: str, permutations: int) -> str:
    scores = [row["cosine_similarity"] for row in rows]
    matched_mean = statistics.mean(scores)
    matched_median = statistics.median(scores)
    matched_stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    mismatch_mean = statistics.mean(mismatch_means)
    mismatch_median = statistics.median(mismatch_means)
    mismatch_stdev = statistics.stdev(mismatch_means) if len(mismatch_means) > 1 else 0.0
    p_value = (sum(value >= matched_mean for value in mismatch_means) + 1) / (len(mismatch_means) + 1)

    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["cosine_similarity"])

    lines = [
        f"Embedding model: {model}",
        f"Pairs analyzed: {len(rows)}",
        f"Permutations: {permutations}",
        "",
        "Matched-pair similarity:",
        f"Mean cosine similarity: {fmt(matched_mean)}",
        f"Median cosine similarity: {fmt(matched_median)}",
        f"Min cosine similarity: {fmt(min(scores))}",
        f"Max cosine similarity: {fmt(max(scores))}",
        f"Std. dev.: {fmt(matched_stdev)}",
        "",
        "Random-mismatch baseline:",
        f"Mean mismatch cosine similarity: {fmt(mismatch_mean)}",
        f"Median mismatch cosine similarity: {fmt(mismatch_median)}",
        f"Std. dev. mismatch mean: {fmt(mismatch_stdev)}",
        "",
        "Matched vs mismatch comparison:",
        f"Mean difference (effect size): {fmt(matched_mean - mismatch_mean)}",
        f"Permutation p-value (matched > mismatched): {fmt(p_value)}",
        "",
        "By category:",
    ]

    for category in sorted(by_category):
        cat_scores = by_category[category]
        lines.append(
            f"{category}: n={len(cat_scores)}, mean={fmt(statistics.mean(cat_scores))}, median={fmt(statistics.median(cat_scores))}"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    client = OpenAI()

    pairs = load_round_1_pairs(args.data)
    sae_vectors = get_embeddings(client, [pair["sae_response"] for pair in pairs], args.model)
    aave_vectors = get_embeddings(client, [pair["aave_response"] for pair in pairs], args.model)

    rows = []
    for pair, sae_vec, aave_vec in zip(pairs, sae_vectors, aave_vectors):
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "category": pair["category"],
                "cosine_similarity": cosine_similarity(sae_vec, aave_vec),
                "sae_response_chars": len(pair["sae_response"]),
                "aave_response_chars": len(pair["aave_response"]),
            }
        )

    mismatch_means = mismatch_distribution(sae_vectors, aave_vectors, args.permutations, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_csv = args.output_dir / "round1_embedding_similarity_results_v3_full.csv"
    mismatch_csv = args.output_dir / "round1_mismatch_distribution_v3_full.csv"
    summary_txt = args.output_dir / "round1_similarity_summary_v3_full.txt"

    write_pair_csv(
        pair_csv,
        [
            {
                **row,
                "cosine_similarity": fmt(row["cosine_similarity"]),
            }
            for row in rows
        ],
    )
    write_mismatch_csv(mismatch_csv, mismatch_means)

    summary_text = build_summary(rows, mismatch_means, args.model, args.permutations)
    summary_txt.write_text(summary_text + "\n")

    print(summary_text)
    print(f"\nPer-pair results written to: {pair_csv}")
    print(f"Mismatch distribution written to: {mismatch_csv}")
    print(f"Summary written to: {summary_txt}")


if __name__ == "__main__":
    main()
