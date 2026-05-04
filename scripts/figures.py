#!/usr/bin/env python3
"""
figures.py — Publication-ready figures for the AAVE LLM study.

Produces separate Round 1 and Round 2 versions of the six main similarity figures:

  1 — Semantic similarity (Cosine)
  2 — Lexical overlap, 1×2 panel (Unigram Jaccard | Bigram Jaccard)
  3 — Per-category semantic similarity (Cosine)
  4 — Per-category lexical overlap, 1×2 panel (Unigram Jaccard | Bigram Jaccard)
  5 — Repeated-run semantic similarity control (Cosine), including matched-trial
      and all-pairs between-dialect comparisons
  6 — Repeated-run lexical overlap control, 1×2 panel (Unigram Jaccard | Bigram
      Jaccard), including matched-trial and all-pairs between-dialect comparisons

Also produces:
  Figure 7 — Horizontal stacked bar: LLM judge verdicts by mitigation condition
             (CLEAN / FLAGGED / DISAGREEMENT, n=15 per condition, legend on side)
             Argument: Fine-Tuned Model 1 + System Prompt had zero flagged cases.

Data sources (relative to repo root — no API calls required):
  Final_Dataset/round1_repeated_runs_similarity.xlsx
  Final_Dataset/round2_repeated_runs_similarity.xlsx
  mitigations/comparison/mitigation_study_logs/*_verdicts.jsonl

Output:
  figures/round1_cosine_similarity.{png,pdf}
  figures/round2_cosine_similarity.{png,pdf}
  figures/round1_jaccard_overlap.{png,pdf}
  figures/round2_jaccard_overlap.{png,pdf}
  figures/round1_cosine_by_category.{png,pdf}
  figures/round2_cosine_by_category.{png,pdf}
  figures/round1_jaccard_by_category.{png,pdf}
  figures/round2_jaccard_by_category.{png,pdf}
  figures/round1_repeated_run_cosine_control.{png,pdf}
  figures/round2_repeated_run_cosine_control.{png,pdf}
  figures/round1_repeated_run_jaccard_control.{png,pdf}
  figures/round2_repeated_run_jaccard_control.{png,pdf}
  figures/judge_verdicts.{png,pdf}

Usage:
    python figures.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO        = Path(__file__).resolve().parent
R1_XLSX     = REPO / "Final_Dataset" / "round1_repeated_runs_similarity.xlsx"
R2_XLSX     = REPO / "Final_Dataset" / "round2_repeated_runs_similarity.xlsx"
VERDICT_DIR = REPO / "mitigations" / "comparison" / "mitigation_study_logs"
OUT_DIR     = REPO / "figures"

# ── Colour palette ─────────────────────────────────────────────────────────────

SAE_COLOR     = "#2166AC"   # blue  — within-SAE
AAVE_COLOR    = "#35978F"   # teal  — within-AAVE
BETWEEN_COLOR = "#D6604D"   # red   — between-dialect (SAE vs AAVE)
ALL_PAIRS_COLOR = "#7B3294"  # purple — between-dialect all-pairs control
CLEAN_COLOR   = "#4DAC26"   # green — no bias detected
FLAGGED_COLOR = "#D6604D"   # red   — bias detected
AMBIG_COLOR   = "#FDB863"   # amber — judge disagreement / unclear


# ── Global matplotlib style ────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


# =============================================================================
# Data loading helpers
# =============================================================================

def _xlsx_rows(path: Path, sheet_name: str) -> list[tuple]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet_name]
    return [tuple(c.value for c in row) for row in ws.iter_rows()]


def load_prompt_cosine(path: Path) -> list[dict]:
    """Read 'Per Prompt (Cosine)' sheet → per-prompt cosine similarity dicts."""
    rows = _xlsx_rows(path, "Per Prompt (Cosine)")
    result = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        result.append({
            "prompt_id":      row[0],
            "category":       row[1],
            "cos_within_sae": float(row[2]),
            "cos_within_aave":float(row[3]),
            "cos_betw_match": float(row[4]),
            "cos_betw_all":   float(row[5]),
        })
    return result


def load_prompt_jaccard(path: Path) -> list[dict]:
    """Read 'Per Prompt (Jaccard)' sheet → per-prompt dicts with unigram and bigram."""
    rows = _xlsx_rows(path, "Per Prompt (Jaccard)")
    result = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        result.append({
            "prompt_id":       row[0],
            "category":        row[1],
            "tok_within_sae":  float(row[2]),
            "tok_within_aave": float(row[3]),
            "tok_betw_match":  float(row[4]),
            "tok_betw_all":    float(row[5]),
            "bi_within_sae":   float(row[6]),
            "bi_within_aave":  float(row[7]),
            "bi_betw_match":   float(row[8]),
            "bi_betw_all":     float(row[9]),
        })
    return result



def load_verdicts() -> dict[str, Counter]:
    """Read all *_verdicts.jsonl files → {condition_name: Counter}."""
    raw: dict[str, Counter] = defaultdict(Counter)
    for vf in sorted(VERDICT_DIR.glob("*_verdicts.jsonl")):
        records = [json.loads(l) for l in vf.read_text().splitlines() if l.strip()]
        if not records:
            continue
        cond = records[0]["condition_name"]
        for rec in records:
            raw[cond][rec["final_verdict"]] += 1
    return dict(raw)


# =============================================================================
# Shared save helper
# =============================================================================

def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Shared helper for grouped bar panels
# =============================================================================

def _grouped_bar_panel(ax, within_sae, within_aave, between, xlabels, ylabel, subtitle,
                       show_legend: bool = True, label_fontsize: float = 9.0):
    """Draw n-group, three-bar grouped bar on ax (works for 2 rounds or 4 categories)."""
    x     = np.arange(len(within_sae), dtype=float)
    width = 0.24
    gap   = 0.03

    pos_sae  = x - width - gap
    pos_aave = x
    pos_betw = x + width + gap

    ax.bar(pos_sae,  within_sae,  width=width, color=SAE_COLOR,
           alpha=0.88, label="Within SAE")
    ax.bar(pos_aave, within_aave, width=width, color=AAVE_COLOR,
           alpha=0.88, label="Within AAVE")
    ax.bar(pos_betw, between,     width=width, color=BETWEEN_COLOR,
           alpha=0.88, label="Between Dialects (SAE vs AAVE)")

    for xi, vals in [(pos_sae, within_sae), (pos_aave, within_aave), (pos_betw, between)]:
        for pos, val in zip(xi, vals):
            ax.text(pos, val / 2, f"{val:.4f}",
                    ha="center", va="center", fontsize=label_fontsize,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, max(list(within_sae) + list(within_aave) + list(between)) + 0.07)
    if show_legend:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


def _repeated_runs_bar_panel(ax, within_sae, within_aave, between_matched,
                             between_all, xlabels, ylabel, subtitle,
                             show_legend: bool = True,
                             label_fontsize: float = 8.5):
    """Draw repeated-run control bars, including matched and all-pairs dialect comparisons."""
    x = np.arange(len(within_sae), dtype=float)
    width = 0.18
    offsets = np.array([-1.65, -0.55, 0.55, 1.65]) * width

    series = [
        (within_sae, "Within SAE", SAE_COLOR),
        (within_aave, "Within AAVE", AAVE_COLOR),
        (between_matched, "Between Dialects (Matched Trials)", BETWEEN_COLOR),
        (between_all, "Between Dialects (All Pairs)", ALL_PAIRS_COLOR),
    ]

    for offset, (values, label, color) in zip(offsets, series):
        positions = x + offset
        ax.bar(positions, values, width=width, color=color, alpha=0.88, label=label)
        for pos, val in zip(positions, values):
            ax.text(pos, val / 2, f"{val:.4f}",
                    ha="center", va="center", fontsize=label_fontsize,
                    color="white", fontweight="bold", rotation=0)

    max_value = max(max(values) for values, _, _ in series)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, max_value + 0.07)
    if show_legend:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


# =============================================================================
# Round-specific similarity figures
# =============================================================================

ROUND_SPECS = [
    {
        "slug": "round1",
        "label": "Round 1",
        "model_label": "Base Model",
        "xlsx": R1_XLSX,
    },
    {
        "slug": "round2",
        "label": "Round 2",
        "model_label": "FT Model + System Prompt",
        "xlsx": R2_XLSX,
    },
]

CATEGORY_ORDER = ["Algorithm", "ELI5", "QA", "Social"]


def _round_xlabel(round_spec: dict, n: int) -> str:
    return f"{round_spec['model_label']}\n{round_spec['label']}  (n = {n} prompts)"


def _ordered_categories(rows: list[dict]) -> list[str]:
    present = {r["category"] for r in rows}
    ordered = [category for category in CATEGORY_ORDER if category in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def plot_round_cosine_similarity(round_spec: dict) -> None:
    """Figure 1 family: one-round cosine similarity summary."""
    rows = load_prompt_cosine(round_spec["xlsx"])
    within_sae = [np.mean([r["cos_within_sae"] for r in rows])]
    within_aave = [np.mean([r["cos_within_aave"] for r in rows])]
    between = [np.mean([r["cos_betw_match"] for r in rows])]
    xlabels = [_round_xlabel(round_spec, len(rows))]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    _grouped_bar_panel(
        ax,
        within_sae,
        within_aave,
        between,
        xlabels,
        ylabel="Cosine Similarity  (mean)",
        subtitle=f"{round_spec['label']} — Semantic Similarity (Cosine)",
        show_legend=False,
    )
    ax.legend(bbox_to_anchor=(0.5, -0.22), loc="upper center",
              ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    stem = f"{round_spec['slug']}_cosine_similarity"
    _save(fig, stem)
    print(f"✓  {round_spec['label']} cosine similarity saved  →  figures/{stem}.{{png,pdf}}")


def plot_round_jaccard_overlap(round_spec: dict) -> None:
    """Figure 2 family: one-round unigram and bigram Jaccard summary."""
    rows = load_prompt_jaccard(round_spec["xlsx"])
    xlabels = [_round_xlabel(round_spec, len(rows))]

    uni_within_sae = [np.mean([r["tok_within_sae"] for r in rows])]
    uni_within_aave = [np.mean([r["tok_within_aave"] for r in rows])]
    uni_between = [np.mean([r["tok_betw_match"] for r in rows])]

    bi_within_sae = [np.mean([r["bi_within_sae"] for r in rows])]
    bi_within_aave = [np.mean([r["bi_within_aave"] for r in rows])]
    bi_between = [np.mean([r["bi_betw_match"] for r in rows])]

    fig, (ax_uni, ax_bi) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    fig.suptitle(
        f"{round_spec['label']} — Jaccard Overlap",
        fontsize=13, fontweight="bold", y=1.01,
    )

    _grouped_bar_panel(
        ax_uni,
        uni_within_sae,
        uni_within_aave,
        uni_between,
        xlabels,
        ylabel="Unigram (Token) Jaccard  (mean)",
        subtitle="Unigram Jaccard\n(individual word overlap)",
        show_legend=False,
        label_fontsize=8.5,
    )
    ax_uni.set_ylim(0, 1.0)

    _grouped_bar_panel(
        ax_bi,
        bi_within_sae,
        bi_within_aave,
        bi_between,
        xlabels,
        ylabel="Bigram Jaccard  (mean)",
        subtitle="Bigram Jaccard\n(consecutive word-pair overlap)",
        show_legend=False,
        label_fontsize=8.5,
    )
    ax_bi.set_ylim(0, 1.0)

    handles, labels = ax_uni.get_legend_handles_labels()
    fig.legend(handles, labels,
               bbox_to_anchor=(0.5, -0.08), loc="upper center",
               ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    stem = f"{round_spec['slug']}_jaccard_overlap"
    _save(fig, stem)
    print(f"✓  {round_spec['label']} Jaccard overlap saved  →  figures/{stem}.{{png,pdf}}")


def plot_round_cosine_by_category(round_spec: dict) -> None:
    """Figure 3 family: one-round per-category cosine similarity."""
    rows = load_prompt_cosine(round_spec["xlsx"])
    order = _ordered_categories(rows)

    by_cat: dict[str, dict[str, list]] = {
        c: {"sae": [], "aave": [], "betw": []} for c in order
    }
    for row in rows:
        c = row["category"]
        by_cat[c]["sae"].append(row["cos_within_sae"])
        by_cat[c]["aave"].append(row["cos_within_aave"])
        by_cat[c]["betw"].append(row["cos_betw_match"])

    within_sae = [np.mean(by_cat[c]["sae"]) for c in order]
    within_aave = [np.mean(by_cat[c]["aave"]) for c in order]
    between = [np.mean(by_cat[c]["betw"]) for c in order]
    ns = [len(by_cat[c]["sae"]) for c in order]
    xlabels = [f"{cat}\n(n = {ns[i]} prompts)" for i, cat in enumerate(order)]

    fig, ax = plt.subplots(figsize=(10, 6.0))
    _grouped_bar_panel(
        ax,
        within_sae,
        within_aave,
        between,
        xlabels,
        ylabel="Cosine Similarity  (mean)",
        subtitle=f"{round_spec['label']} — Per-Category Semantic Similarity (Cosine)",
        show_legend=False,
        label_fontsize=7.0 if len(order) > 3 else 8.0,
    )
    ax.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center",
              ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    stem = f"{round_spec['slug']}_cosine_by_category"
    _save(fig, stem)
    print(f"✓  {round_spec['label']} cosine by category saved  →  figures/{stem}.{{png,pdf}}")


def plot_round_jaccard_by_category(round_spec: dict) -> None:
    """Figure 4 family: one-round per-category unigram and bigram Jaccard."""
    rows = load_prompt_jaccard(round_spec["xlsx"])
    order = _ordered_categories(rows)

    by_cat: dict[str, dict[str, list]] = {
        c: {
            "tok_sae": [],
            "tok_aave": [],
            "tok_betw": [],
            "bi_sae": [],
            "bi_aave": [],
            "bi_betw": [],
        }
        for c in order
    }
    for row in rows:
        c = row["category"]
        by_cat[c]["tok_sae"].append(row["tok_within_sae"])
        by_cat[c]["tok_aave"].append(row["tok_within_aave"])
        by_cat[c]["tok_betw"].append(row["tok_betw_match"])
        by_cat[c]["bi_sae"].append(row["bi_within_sae"])
        by_cat[c]["bi_aave"].append(row["bi_within_aave"])
        by_cat[c]["bi_betw"].append(row["bi_betw_match"])

    ns = [len(by_cat[c]["tok_sae"]) for c in order]
    xlabels = [f"{cat}\n(n = {ns[i]} prompts)" for i, cat in enumerate(order)]

    uni_sae = [np.mean(by_cat[c]["tok_sae"]) for c in order]
    uni_aave = [np.mean(by_cat[c]["tok_aave"]) for c in order]
    uni_betw = [np.mean(by_cat[c]["tok_betw"]) for c in order]

    bi_sae = [np.mean(by_cat[c]["bi_sae"]) for c in order]
    bi_aave = [np.mean(by_cat[c]["bi_aave"]) for c in order]
    bi_betw = [np.mean(by_cat[c]["bi_betw"]) for c in order]

    fig, (ax_uni, ax_bi) = plt.subplots(1, 2, figsize=(15, 6.0))
    fig.suptitle(
        f"{round_spec['label']} — Per-Category Jaccard Overlap",
        fontsize=13, fontweight="bold", y=1.01,
    )

    _grouped_bar_panel(
        ax_uni,
        uni_sae,
        uni_aave,
        uni_betw,
        xlabels,
        ylabel="Unigram (Token) Jaccard  (mean)",
        subtitle="Unigram Jaccard\n(individual word overlap)",
        show_legend=False,
        label_fontsize=7.0,
    )
    ax_uni.set_ylim(0, 1.0)

    _grouped_bar_panel(
        ax_bi,
        bi_sae,
        bi_aave,
        bi_betw,
        xlabels,
        ylabel="Bigram Jaccard  (mean)",
        subtitle="Bigram Jaccard\n(consecutive word-pair overlap)",
        show_legend=False,
        label_fontsize=7.0,
    )
    ax_bi.set_ylim(0, 1.0)

    handles, labels = ax_uni.get_legend_handles_labels()
    fig.legend(handles, labels,
               bbox_to_anchor=(0.5, -0.08), loc="upper center",
               ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    stem = f"{round_spec['slug']}_jaccard_by_category"
    _save(fig, stem)
    print(f"✓  {round_spec['label']} Jaccard by category saved  →  figures/{stem}.{{png,pdf}}")


def plot_round_repeated_run_cosine_control(round_spec: dict) -> None:
    """Figure 5 family: one-round repeated-run cosine control."""
    rows = load_prompt_cosine(round_spec["xlsx"])
    within_sae = [np.mean([r["cos_within_sae"] for r in rows])]
    within_aave = [np.mean([r["cos_within_aave"] for r in rows])]
    between_matched = [np.mean([r["cos_betw_match"] for r in rows])]
    between_all = [np.mean([r["cos_betw_all"] for r in rows])]
    xlabels = [_round_xlabel(round_spec, len(rows))]

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    _repeated_runs_bar_panel(
        ax,
        within_sae,
        within_aave,
        between_matched,
        between_all,
        xlabels,
        ylabel="Cosine Similarity  (mean)",
        subtitle=f"{round_spec['label']} — Repeated-Run Semantic Similarity Control (Cosine)",
        show_legend=False,
        label_fontsize=8.0,
    )
    ax.legend(bbox_to_anchor=(0.5, -0.22), loc="upper center",
              ncol=2, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    stem = f"{round_spec['slug']}_repeated_run_cosine_control"
    _save(fig, stem)
    print(f"✓  {round_spec['label']} repeated-run cosine control saved  →  figures/{stem}.{{png,pdf}}")


def plot_round_repeated_run_jaccard_control(round_spec: dict) -> None:
    """Figure 6 family: one-round repeated-run unigram and bigram Jaccard control."""
    rows = load_prompt_jaccard(round_spec["xlsx"])
    xlabels = [_round_xlabel(round_spec, len(rows))]

    uni_within_sae = [np.mean([r["tok_within_sae"] for r in rows])]
    uni_within_aave = [np.mean([r["tok_within_aave"] for r in rows])]
    uni_between_matched = [np.mean([r["tok_betw_match"] for r in rows])]
    uni_between_all = [np.mean([r["tok_betw_all"] for r in rows])]

    bi_within_sae = [np.mean([r["bi_within_sae"] for r in rows])]
    bi_within_aave = [np.mean([r["bi_within_aave"] for r in rows])]
    bi_between_matched = [np.mean([r["bi_betw_match"] for r in rows])]
    bi_between_all = [np.mean([r["bi_betw_all"] for r in rows])]

    fig, (ax_uni, ax_bi) = plt.subplots(1, 2, figsize=(14.5, 6.0))
    fig.suptitle(
        f"{round_spec['label']} — Repeated-Run Jaccard Overlap Control",
        fontsize=13, fontweight="bold", y=1.01,
    )

    _repeated_runs_bar_panel(
        ax_uni,
        uni_within_sae,
        uni_within_aave,
        uni_between_matched,
        uni_between_all,
        xlabels,
        ylabel="Unigram (Token) Jaccard  (mean)",
        subtitle="Unigram Jaccard\n(individual word overlap)",
        show_legend=False,
        label_fontsize=7.2,
    )
    ax_uni.set_ylim(0, 1.0)

    _repeated_runs_bar_panel(
        ax_bi,
        bi_within_sae,
        bi_within_aave,
        bi_between_matched,
        bi_between_all,
        xlabels,
        ylabel="Bigram Jaccard  (mean)",
        subtitle="Bigram Jaccard\n(consecutive word-pair overlap)",
        show_legend=False,
        label_fontsize=7.2,
    )
    ax_bi.set_ylim(0, 1.0)

    handles, labels = ax_uni.get_legend_handles_labels()
    fig.legend(handles, labels,
               bbox_to_anchor=(0.5, -0.08), loc="upper center",
               ncol=4, fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    stem = f"{round_spec['slug']}_repeated_run_jaccard_control"
    _save(fig, stem)
    print(f"✓  {round_spec['label']} repeated-run Jaccard control saved  →  figures/{stem}.{{png,pdf}}")


def generate_round_similarity_figures(round_spec: dict) -> None:
    """Generate all six round-specific similarity figures for one study round."""
    print(f"\nGenerating {round_spec['label']} similarity figures...")
    plot_round_cosine_similarity(round_spec)
    plot_round_jaccard_overlap(round_spec)
    plot_round_cosine_by_category(round_spec)
    plot_round_jaccard_by_category(round_spec)
    plot_round_repeated_run_cosine_control(round_spec)
    plot_round_repeated_run_jaccard_control(round_spec)


# =============================================================================
# Figure 7 — Stacked Bar: LLM Judge Verdicts by Mitigation Condition
# =============================================================================

def figure7() -> None:
    """
    Horizontal stacked bar — one bar per mitigation condition (5 bars total).
    The two Base Model runs are merged into a single bar.
    All conditions normalised to n=15.
    Legend placed outside the plot on the right.
    """
    verdicts = load_verdicts()

    # Merge the two base-model runs into one condition
    merged_base: Counter = Counter()
    merged_base.update(verdicts.get("base_sysprompt_run1", Counter()))
    merged_base.update(verdicts.get("base_sysprompt_run2", Counter()))
    base_total = sum(merged_base.values())
    base_factor = base_total // 15
    verdicts["base_sysprompt"] = Counter(
        {k: v // base_factor for k, v in merged_base.items()}
    )

    # Normalise any other condition accidentally run twice (n=30 → n=15)
    for cond in list(verdicts.keys()):
        if cond in ("base_sysprompt", "base_sysprompt_run1", "base_sysprompt_run2"):
            continue
        total = sum(verdicts[cond].values())
        if total != 15 and total % 15 == 0:
            factor = total // 15
            verdicts[cond] = Counter(
                {k: v // factor for k, v in verdicts[cond].items()}
            )

    display_order = [
        "base_sysprompt",
        "ft_model_1_only",
        "ft_model_1_with_sysprompt",
        "ft_model_2_only",
        "ft_model_2_with_sysprompt",
    ]
    label_map = {
        "base_sysprompt":            "Base Model + System Prompt",
        "ft_model_1_only":           "Fine-Tuned Model 1",
        "ft_model_1_with_sysprompt": "Fine-Tuned Model 1 + System Prompt",
        "ft_model_2_only":           "Fine-Tuned Model 2",
        "ft_model_2_with_sysprompt": "Fine-Tuned Model 2 + System Prompt",
    }

    conditions   = [c for c in display_order if c in verdicts]
    labels       = [label_map[c] for c in conditions]
    totals       = [sum(verdicts[c].values()) for c in conditions]

    clean_pct    = [verdicts[c].get("CLEAN",        0) / t * 100 for c, t in zip(conditions, totals)]
    flagged_pct  = [verdicts[c].get("FLAGGED",      0) / t * 100 for c, t in zip(conditions, totals)]
    disagree_pct = [verdicts[c].get("DISAGREEMENT", 0) / t * 100 for c, t in zip(conditions, totals)]
    clean_n      = [verdicts[c].get("CLEAN",        0) for c in conditions]
    flagged_n    = [verdicts[c].get("FLAGGED",      0) for c in conditions]
    disagree_n   = [verdicts[c].get("DISAGREEMENT", 0) for c in conditions]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    y      = np.arange(len(conditions))
    height = 0.50

    ax.barh(y, clean_pct, height,
            color=CLEAN_COLOR, alpha=0.85,
            label="CLEAN — no user-attribution bias")
    ax.barh(y, flagged_pct, height, left=clean_pct,
            color=FLAGGED_COLOR, alpha=0.85,
            label="FLAGGED — bias detected in AAVE response")

    left_for_disagree = [c + f for c, f in zip(clean_pct, flagged_pct)]
    ax.barh(y, disagree_pct, height, left=left_for_disagree,
            color=AMBIG_COLOR, alpha=0.85,
            label="DISAGREEMENT — judges disagreed")

    # Count labels inside bars
    for i in range(len(conditions)):
        total = totals[i]
        if clean_pct[i] > 7:
            ax.text(clean_pct[i] / 2, y[i],
                    f"{clean_n[i]}/{total}",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        if flagged_pct[i] > 3.5:
            ax.text(clean_pct[i] + flagged_pct[i] / 2, y[i],
                    str(flagged_n[i]),
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
        if disagree_pct[i] > 3.5:
            ax.text(left_for_disagree[i] + disagree_pct[i] / 2, y[i],
                    str(disagree_n[i]),
                    ha="center", va="center", fontsize=9,
                    color="#333333", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()

    ax.set_xlabel("Proportion of Prompt Evaluations (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title(
        "Figure 7 — LLM Judge Verdicts by Mitigation Condition",
        fontsize=12, fontweight="bold", pad=12,
    )

    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left",
              fontsize=9.5, framealpha=0.9)

    fig.tight_layout()

    _save(fig, "judge_verdicts")
    print("✓  Figure 7 saved  →  figures/judge_verdicts.{png,pdf}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    for round_spec in ROUND_SPECS:
        generate_round_similarity_figures(round_spec)

    print("\nGenerating Figure 7 — Judge Verdicts by Condition...")
    figure7()

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
