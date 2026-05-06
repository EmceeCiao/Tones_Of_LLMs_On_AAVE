#!/usr/bin/env python3
"""
Round 1 human-evaluation figures from Qualtrics data.

Reads:
  Round_1_Results/Qualtrics_Data/
    LLM & AAVE ROUND 1_April 29, 2026_02.09_aggregated_simple_summary.xlsx

Writes:
  qualtrics_figures/round1/figures/round1_preference_by_category.{png,pdf}
  qualtrics_figures/round1/figures/round1_preference_margin_by_prompt.{png,pdf}
  qualtrics_figures/round1/figures/round1_tone_gap_by_prompt.{png,pdf}
  qualtrics_figures/round1/figures/round1_rating_difference_heatmap.{png,pdf}
  qualtrics_figures/round1/figures/round1_tone_gap_by_category.{png,pdf}

Usage:
  python qualtrics_figures/round1/qualtrics_figures_round1.py
"""

from __future__ import annotations

from pathlib import Path
from textwrap import shorten

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl


REPO = Path(__file__).resolve().parent.parent.parent
INPUT_XLSX = (
    REPO
    / "Round_1_Results"
    / "Qualtrics_Data"
    / "LLM & AAVE ROUND 1_April 29, 2026_02.09_aggregated_simple_summary.xlsx"
)
OUT_DIR = REPO / "qualtrics_figures" / "round1" / "figures"


SAE_COLOR = "#2166AC"
AAVE_COLOR = "#35978F"
NO_PREF_COLOR = "#BDBDBD"
NEUTRAL_COLOR = "#555555"
POS_COLOR = "#2166AC"
NEG_COLOR = "#D6604D"


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _load_sheet(sheet_name: str) -> list[dict]:
    wb = openpyxl.load_workbook(INPUT_XLSX, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(value) for value in rows[0]]
    return [
        {header: value for header, value in zip(headers, row)}
        for row in rows[1:]
        if row and row[0] is not None
    ]


def _category_order(rows: list[dict]) -> list[str]:
    preferred = ["Algorithm", "ELI5", "QA", "Social"]
    present = {row["Category"] for row in rows}
    ordered = [category for category in preferred if category in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _prompt_label(row: dict) -> str:
    prompt_id = row["PromptID"]
    category = row["Category"]
    prompt = shorten(str(row["Prompt"]).replace("\n", " "), width=44, placeholder="...")
    return f"{prompt_id} ({category})\n{prompt}"


def preference_by_category(category_rows: list[dict]) -> None:
    """Stacked percentage bars showing SAE/AAVE/no-preference votes by category."""
    rows = sorted(category_rows, key=lambda row: _category_order(category_rows).index(row["Category"]))
    categories = [row["Category"] for row in rows]
    sae = np.array([row["Preference_SAE_Count"] for row in rows], dtype=float)
    aave = np.array([row["Preference_AAVE_Count"] for row in rows], dtype=float)
    no_pref = np.array([row["Preference_No_Preference_Count"] for row in rows], dtype=float)
    totals = sae + aave + no_pref

    sae_pct = sae / totals * 100
    aave_pct = aave / totals * 100
    no_pref_pct = no_pref / totals * 100

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(categories))
    height = 0.58

    ax.barh(y, sae_pct, height, color=SAE_COLOR, alpha=0.88, label="SAE preferred")
    ax.barh(y, aave_pct, height, left=sae_pct, color=AAVE_COLOR, alpha=0.88, label="AAVE preferred")
    ax.barh(
        y,
        no_pref_pct,
        height,
        left=sae_pct + aave_pct,
        color=NO_PREF_COLOR,
        alpha=0.88,
        label="No preference",
    )

    for i, total in enumerate(totals):
        if sae_pct[i] > 8:
            ax.text(sae_pct[i] / 2, y[i], f"{int(sae[i])}/{int(total)}",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if aave_pct[i] > 8:
            ax.text(sae_pct[i] + aave_pct[i] / 2, y[i], f"{int(aave[i])}/{int(total)}",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if no_pref_pct[i] > 8:
            ax.text(sae_pct[i] + aave_pct[i] + no_pref_pct[i] / 2, y[i],
                    f"{int(no_pref[i])}/{int(total)}",
                    ha="center", va="center", color="#333333", fontweight="bold", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlabel("Preference Votes (%)")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.set_title("Round 1 — Human Preference Distribution by Category",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center",
              ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, "round1_preference_by_category")


def preference_margin_by_prompt(prompt_rows: list[dict]) -> None:
    """Diverging prompt-level SAE-vs-AAVE preference margin."""
    rows = sorted(
        prompt_rows,
        key=lambda row: row["Preference_SAE_Count"] - row["Preference_AAVE_Count"],
    )
    labels = [_prompt_label(row) for row in rows]
    margins = np.array([
        row["Preference_SAE_Count"] - row["Preference_AAVE_Count"]
        for row in rows
    ], dtype=float)
    colors = [POS_COLOR if margin >= 0 else NEG_COLOR for margin in margins]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.barh(y, margins, color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=1.1)

    for pos, margin in zip(y, margins):
        ha = "left" if margin >= 0 else "right"
        dx = 0.12 if margin >= 0 else -0.12
        ax.text(margin + dx, pos, f"{int(margin):+d}",
                ha=ha, va="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("Preference Margin  (SAE votes - AAVE votes)")
    ax.set_title("Round 1 — Per-Prompt Human Preference Margin",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(min(margins) - 1.1, max(margins) + 1.1)
    fig.tight_layout()
    _save(fig, "round1_preference_margin_by_prompt")


def tone_gap_by_prompt(prompt_rows: list[dict]) -> None:
    """Diverging prompt-level SAE-minus-AAVE tone gap."""
    rows = sorted(prompt_rows, key=lambda row: row["Tone_SAE_minus_AAVE"])
    labels = [_prompt_label(row) for row in rows]
    gaps = np.array([row["Tone_SAE_minus_AAVE"] for row in rows], dtype=float)
    colors = [POS_COLOR if gap >= 0 else NEG_COLOR for gap in gaps]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.barh(y, gaps, color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=1.1)

    for pos, gap in zip(y, gaps):
        ha = "left" if gap >= 0 else "right"
        dx = 0.018 if gap >= 0 else -0.018
        ax.text(gap + dx, pos, f"{gap:+.3f}",
                ha=ha, va="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("Tone Gap  (SAE mean - AAVE mean)")
    ax.set_title("Round 1 — Per-Prompt Tone Rating Gap",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(min(gaps) - 0.12, max(gaps) + 0.12)
    fig.tight_layout()
    _save(fig, "round1_tone_gap_by_prompt")


def rating_difference_heatmap(prompt_rows: list[dict]) -> None:
    """Prompt-by-metric heatmap of SAE-minus-AAVE mean rating differences."""
    rows = sorted(prompt_rows, key=lambda row: row["PromptID"])
    metrics = ["Helpful", "Clarity", "Warmth", "Tone"]
    matrix = np.array([
        [
            row[f"SAE_{metric}_Mean"] - row[f"AAVE_{metric}_Mean"]
            for metric in metrics
        ]
        for row in rows
    ], dtype=float)

    vlim = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 0.1)
    labels = [f"{row['PromptID']} ({row['Category']})" for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Round 1 — Rating Difference Heatmap (SAE - AAVE)",
                 fontsize=13, fontweight="bold", pad=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if abs(value) > vlim * 0.55 else "#222222"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SAE mean - AAVE mean")
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "round1_rating_difference_heatmap")


def tone_gap_by_category(category_rows: list[dict]) -> None:
    """Category-level SAE-minus-AAVE tone gap."""
    rows = sorted(category_rows, key=lambda row: _category_order(category_rows).index(row["Category"]))
    categories = [row["Category"] for row in rows]
    gaps = np.array([row["Tone_SAE_minus_AAVE"] for row in rows], dtype=float)
    colors = [POS_COLOR if gap >= 0 else NEG_COLOR for gap in gaps]
    x = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x, gaps, color=colors, alpha=0.88)
    ax.axhline(0, color="#333333", linewidth=1.1)

    for pos, gap in zip(x, gaps):
        va = "bottom" if gap >= 0 else "top"
        dy = 0.012 if gap >= 0 else -0.012
        ax.text(pos, gap + dy, f"{gap:+.3f}",
                ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Tone Gap  (SAE mean - AAVE mean)")
    ax.set_title("Round 1 — Category-Level Tone Rating Gap",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(min(gaps) - 0.08, max(gaps) + 0.08)
    fig.tight_layout()
    _save(fig, "round1_tone_gap_by_category")


def main() -> None:
    prompt_rows = _load_sheet("PromptSummary")
    category_rows = _load_sheet("CategorySummary")

    preference_by_category(category_rows)
    preference_margin_by_prompt(prompt_rows)
    tone_gap_by_prompt(prompt_rows)
    rating_difference_heatmap(prompt_rows)
    tone_gap_by_category(category_rows)

    print(f"Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
