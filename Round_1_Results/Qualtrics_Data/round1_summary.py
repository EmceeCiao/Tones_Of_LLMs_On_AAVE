#!/usr/bin/env python3
"""Simple Round 1 summary from the aggregated Qualtrics workbook.

It only computes:

- average tone scores by prompt and category
- preference tallies and majority
- whether the exact higher tone average agrees with the preference majority
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path

from aggregate_round1_qualtrics import dense_row, read_xlsx_first_sheet, write_xlsx


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_AGGREGATED = (
    SCRIPT_DIR / "LLM & AAVE ROUND 1_April 29, 2026_02.09_aggregated.xlsx"
)
DEFAULT_PROMPT_MAP = REPO_ROOT / "Final_Dataset" / "Qualtrics" / "Round_1" / "Round_1_Qualtrics.json"
DEFAULT_OUTPUT = DEFAULT_AGGREGATED.with_name(DEFAULT_AGGREGATED.stem + "_simple_summary.xlsx")

SCORE_METRICS = ("Helpful", "Clarity", "Warmth")
DIALECTS = ("SAE", "AAVE")
PREFERENCE_LABELS = ("SAE", "AAVE", "No Preference")


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def read_table(path: Path) -> list[dict[str, str]]:
    rows = read_xlsx_first_sheet(path)
    width = max(max(row.keys(), default=0) for row in rows)
    headers = dense_row(rows[0], width)
    output = []
    for row in rows[1:]:
        values = dense_row(row, width)
        output.append({header: values[index] for index, header in enumerate(headers) if header})
    return output


def load_prompt_map(path: Path) -> dict[str, dict[str, str]]:
    items = json.loads(path.read_text())
    prompt_map = {}
    for index, item in enumerate(items, start=1):
        prompt_id = f"P{index:02d}"
        prompt_map[prompt_id] = {
            "category": item.get("category", ""),
            "prompt": item.get("prompt") or item.get("sae_prompt", ""),
        }
    return prompt_map


def prompt_ids_from_rows(rows: list[dict[str, str]]) -> list[str]:
    prompt_ids = set()
    for row in rows:
        for header in row:
            match = re.match(r"^(P\d+)_", header)
            if match:
                prompt_ids.add(match.group(1))
    return sorted(prompt_ids)


def normalize_preference(value: str) -> str:
    value = value.strip()
    if value.lower() == "no preference":
        return "No Preference"
    return value


def majority_label(counts: Counter[str]) -> str:
    highest = max((counts[label] for label in PREFERENCE_LABELS), default=0)
    if highest == 0:
        return ""
    winners = [label for label in PREFERENCE_LABELS if counts[label] == highest]
    if len(winners) == 1:
        return winners[0]
    return "Tie: " + "; ".join(winners)


def tone_winner(sae_tone: float | None, aave_tone: float | None) -> str:
    if sae_tone is None or aave_tone is None:
        return ""
    if sae_tone == aave_tone:
        return "Tie"
    return "SAE" if sae_tone > aave_tone else "AAVE"


def agreement(preference_majority: str, winner: str) -> str:
    if not preference_majority or not winner:
        return ""
    if preference_majority in ("SAE", "AAVE"):
        return "Agree" if preference_majority == winner else "Disagree"
    if preference_majority == "No Preference":
        return "Agree" if winner == "Tie" else "Disagree"
    return "Tie/Mixed"


def build_prompt_summary(
    rows: list[dict[str, str]], prompt_map: dict[str, dict[str, str]], prompt_ids: list[str]
) -> tuple[list[list[str]], list[dict[str, object]]]:
    headers = [
        "PromptID",
        "Category",
        "Prompt",
        "N_Preference_Votes",
        "SAE_Helpful_Mean",
        "SAE_Clarity_Mean",
        "SAE_Warmth_Mean",
        "SAE_Tone_Mean",
        "AAVE_Helpful_Mean",
        "AAVE_Clarity_Mean",
        "AAVE_Warmth_Mean",
        "AAVE_Tone_Mean",
        "Preference_SAE_Count",
        "Preference_AAVE_Count",
        "Preference_No_Preference_Count",
        "Preference_Majority",
        "Tone_Winner",
        "Tone_SAE_minus_AAVE",
        "Preference_vs_Tone",
    ]
    output = [headers]
    stats: list[dict[str, object]] = []

    for prompt_id in prompt_ids:
        scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        preferences: Counter[str] = Counter()
        for row in rows:
            preference = normalize_preference(row.get(f"{prompt_id}_Preference", ""))
            if preference:
                preferences[preference] += 1
            for dialect in DIALECTS:
                for metric in SCORE_METRICS:
                    value = parse_float(row.get(f"{prompt_id}_{dialect}_{metric}", ""))
                    if value is not None:
                        scores[(dialect, metric)].append(value)

        metric_means = {
            (dialect, metric): mean(scores[(dialect, metric)])
            for dialect in DIALECTS
            for metric in SCORE_METRICS
        }
        tone_means = {
            dialect: mean(
                [
                    value
                    for metric in SCORE_METRICS
                    for value in scores[(dialect, metric)]
                ]
            )
            for dialect in DIALECTS
        }
        pref_majority = majority_label(preferences)
        winner = tone_winner(tone_means["SAE"], tone_means["AAVE"])
        diff = (
            None
            if tone_means["SAE"] is None or tone_means["AAVE"] is None
            else tone_means["SAE"] - tone_means["AAVE"]
        )
        prompt_info = prompt_map.get(prompt_id, {})

        output.append(
            [
                prompt_id,
                prompt_info.get("category", ""),
                prompt_info.get("prompt", ""),
                str(sum(preferences.values())),
                fmt(metric_means[("SAE", "Helpful")]),
                fmt(metric_means[("SAE", "Clarity")]),
                fmt(metric_means[("SAE", "Warmth")]),
                fmt(tone_means["SAE"]),
                fmt(metric_means[("AAVE", "Helpful")]),
                fmt(metric_means[("AAVE", "Clarity")]),
                fmt(metric_means[("AAVE", "Warmth")]),
                fmt(tone_means["AAVE"]),
                str(preferences["SAE"]),
                str(preferences["AAVE"]),
                str(preferences["No Preference"]),
                pref_majority,
                winner,
                fmt(diff),
                agreement(pref_majority, winner),
            ]
        )
        stats.append(
            {
                "prompt_id": prompt_id,
                "category": prompt_info.get("category", ""),
                "scores": scores,
                "preferences": preferences,
            }
        )

    return output, stats


def build_category_summary(prompt_stats: list[dict[str, object]]) -> list[list[str]]:
    headers = [
        "Category",
        "N_Prompts",
        "N_Preference_Votes",
        "SAE_Tone_Mean",
        "AAVE_Tone_Mean",
        "Preference_SAE_Count",
        "Preference_AAVE_Count",
        "Preference_No_Preference_Count",
        "Preference_Majority",
        "Tone_Winner",
        "Tone_SAE_minus_AAVE",
        "Preference_vs_Tone",
    ]
    output = [headers]
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for stat in prompt_stats:
        by_category[str(stat["category"])].append(stat)

    for category in sorted(by_category):
        stats = by_category[category]
        preferences: Counter[str] = Counter()
        scores: dict[str, list[float]] = defaultdict(list)
        for stat in stats:
            preferences.update(stat["preferences"])
            for dialect in DIALECTS:
                for metric in SCORE_METRICS:
                    scores[dialect].extend(stat["scores"][(dialect, metric)])

        sae_tone = mean(scores["SAE"])
        aave_tone = mean(scores["AAVE"])
        pref_majority = majority_label(preferences)
        winner = tone_winner(sae_tone, aave_tone)
        diff = None if sae_tone is None or aave_tone is None else sae_tone - aave_tone
        output.append(
            [
                category,
                str(len(stats)),
                str(sum(preferences.values())),
                fmt(sae_tone),
                fmt(aave_tone),
                str(preferences["SAE"]),
                str(preferences["AAVE"]),
                str(preferences["No Preference"]),
                pref_majority,
                winner,
                fmt(diff),
                agreement(pref_majority, winner),
            ]
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated", type=Path, default=DEFAULT_AGGREGATED)
    parser.add_argument("--prompt-map", type=Path, default=DEFAULT_PROMPT_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_table(args.aggregated)
    prompt_map = load_prompt_map(args.prompt_map)
    prompt_ids = prompt_ids_from_rows(rows)
    prompt_summary, prompt_stats = build_prompt_summary(rows, prompt_map, prompt_ids)
    category_summary = build_category_summary(prompt_stats)
    write_xlsx(args.output, [("PromptSummary", prompt_summary), ("CategorySummary", category_summary)])
    print(f"Wrote {args.output}")
    print(f"Prompts summarized: {len(prompt_summary) - 1}")
    print(f"Categories summarized: {len(category_summary) - 1}")


if __name__ == "__main__":
    main()
