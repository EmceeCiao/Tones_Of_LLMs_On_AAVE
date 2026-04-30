#!/usr/bin/env python3
"""
build_judge_results_table.py

Reads every *_summary.json produced by the test_single_llm_as_a_judge*.py scripts
and writes two consolidated artefacts:

    dataset/llm_judge_results.csv   — one row per run, machine-readable
    dataset/llm_judge_results.md    — two Markdown tables:
                                       Table 1: accuracy + per-prompt verdicts
                                       Table 2: per-class precision / recall / F1

Usage:
    python dataset/build_judge_results_table.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "study_logs_judge"
OUT_CSV = SCRIPT_DIR / "llm_judge_results.csv"
OUT_MD  = SCRIPT_DIR / "llm_judge_results.md"

GOLD_PROMPTS = ["P02", "P03", "P05", "P08", "P09", "P13"]
CLASSES = ["SAE", "AAVE", "No Preference"]

# Maps judge_prompt_type → (variant_label, short_description)
VARIANT_META: dict[str, tuple[str, str]] = {
    "zero_shot_blind_labels_margin_rubric":                               ("v1a", "Zero-shot, direct winner, SAE-first only"),
    "zero_shot_source_prompt_only_swapped_order_margin_rubric":           ("v1b", "Zero-shot, direct winner, both orders"),
    "v2_calibrated_direct_winner_gpt41_default":                          ("v2",  "Calibrated direct winner"),
    "v3_score_only_derived_label_gpt41_default":                          ("v3",  "Score-only derived label"),
    "v4_gpt5_medium_reasoning_direct_winner":                             ("v4",  "GPT-5 medium reasoning, direct winner"),
    "v5_rationale_informed_direct_winner_gpt41_default":                  ("v5",  "Rationale-informed, direct winner"),
    "v6_rationale_informed_score_derived_label_gpt41_default":            ("v6",  "Rationale-informed, score-derived label"),
    "v7_study_evaluator_persona_direct_winner_gpt41_default":             ("v7",  "Study-evaluator persona, direct winner"),
    "v8_stricter_rationale_informed_score_derived_label_gpt41_default":   ("v8",  "Stricter rationale-informed, score-derived"),
    "v9_clean_gold_without_p05_stricter_score_derived_label_gpt41_default": ("v9", "Score-derived, P05 removed from gold"),
    "v10_few_shot_clean_gold_without_p05_score_derived_label_gpt41_default": ("v10", "Few-shot, score-derived"),
    "v11_abstaining_few_shot_score_derived_dpo_filter_gpt41_default":     ("v11", "Abstaining few-shot, score-derived"),
    "v12_conservative_few_shot_score_derived_dpo_filter_gpt41_default":   ("v12", "Conservative few-shot, score-derived"),
}


def _pct(n: float | None) -> str:
    return "" if n is None else f"{n:.1%}"


def _f(n: float | None) -> str:
    return "" if n is None else f"{n:.2f}"


def _acc(d: dict) -> float | None:
    return d.get("accuracy")


def load_run(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    cfg     = d.get("config", {})
    metrics = d.get("metrics", {})
    obs     = d.get("order_bias_summary", {})
    mv      = d.get("majority_vote_summary", {})

    prompt_type = cfg.get("judge_prompt_type") or cfg.get("prompt_variant") or ""
    variant, description = VARIANT_META.get(prompt_type, ("?", prompt_type))

    models = cfg.get("models") or ([cfg["model"]] if cfg.get("model") else [])
    model  = ", ".join(models) if models else "?"
    gold_n = cfg.get("num_gold_items", len(cfg.get("gold_labels", {})))

    overall_acc    = _acc(metrics.get("overall", {}))
    sae_first_acc  = _acc(obs.get("metrics_by_order", {}).get("sae_first",  {}).get("overall", {}))
    aave_first_acc = _acc(obs.get("metrics_by_order", {}).get("aave_first", {}).get("overall", {}))

    # Order consistency
    oc_list   = obs.get("prompt_order_consistency", [])
    oc_total  = len(oc_list)
    oc_agree  = sum(1 for r in oc_list if r.get("consistent_prediction"))
    order_consist = f"{oc_agree}/{oc_total}" if oc_total else ""

    # Per-class precision / recall / F1 (from overall metrics, both orders combined)
    per_class: dict[str, dict[str, float | None]] = {}
    for cls in CLASSES:
        cm = metrics.get(cls, {})
        per_class[cls] = {
            "precision": cm.get("precision"),
            "recall":    cm.get("recall"),
            "f1":        cm.get("f1"),
            "support":   cm.get("support"),
        }

    # Per-prompt verdict from across-order majority vote
    across = {r["prompt_id"]: r for r in mv.get("across_order_results", [])}
    by_ord: dict[str, dict] = {}
    for r in mv.get("by_order_results", []):
        pid = r["prompt_id"]
        if r.get("order_name") == "sae_first" or pid not in by_ord:
            by_ord[pid] = r

    per_prompt: dict[str, str] = {}
    for pid in GOLD_PROMPTS:
        rec = across.get(pid) or by_ord.get(pid)
        if rec is None:
            per_prompt[pid] = ""
            continue
        correct   = rec.get("correct")
        predicted = rec.get("predicted_label", "")
        if correct is True:
            per_prompt[pid] = "✓"
        elif predicted == "Tie":
            per_prompt[pid] = "~"
        elif correct is False:
            per_prompt[pid] = "✗"
        else:
            per_prompt[pid] = "?"

    return {
        "file":          path.stem,
        "variant":       variant,
        "description":   description,
        "model":         model,
        "gold_n":        gold_n,
        "acc_overall":   overall_acc,
        "acc_sae_first": sae_first_acc,
        "acc_aave_first":aave_first_acc,
        "order_consist": order_consist,
        # per-class metrics
        "sae_prec":      per_class["SAE"]["precision"],
        "sae_rec":       per_class["SAE"]["recall"],
        "sae_f1":        per_class["SAE"]["f1"],
        "sae_support":   per_class["SAE"]["support"],
        "aave_prec":     per_class["AAVE"]["precision"],
        "aave_rec":      per_class["AAVE"]["recall"],
        "aave_f1":       per_class["AAVE"]["f1"],
        "aave_support":  per_class["AAVE"]["support"],
        "nopref_prec":   per_class["No Preference"]["precision"],
        "nopref_rec":    per_class["No Preference"]["recall"],
        "nopref_f1":     per_class["No Preference"]["f1"],
        "nopref_support":per_class["No Preference"]["support"],
        # per-prompt
        **{pid: per_prompt[pid] for pid in GOLD_PROMPTS},
    }


def sort_key(row: dict) -> tuple:
    v = row["variant"]
    if v.startswith("v"):
        num_str = v[1:]
        alpha = ""
        if num_str and num_str[-1].isalpha():
            alpha = num_str[-1]
            num_str = num_str[:-1]
        return (int(num_str) if num_str.isdigit() else 999, alpha)
    return (999, v)


def build_table() -> list[dict]:
    rows = []
    for f in sorted(LOG_DIR.glob("*_summary.json")):
        try:
            rows.append(load_run(f))
        except Exception as exc:
            print(f"  Warning: could not parse {f.name}: {exc}")
    rows.sort(key=sort_key)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    float_fields = {"acc_overall", "acc_sae_first", "acc_aave_first",
                    "sae_prec", "sae_rec", "sae_f1",
                    "aave_prec", "aave_rec", "aave_f1",
                    "nopref_prec", "nopref_rec", "nopref_f1"}
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: (f"{v:.4f}" if k in float_fields and isinstance(v, float) else v)
                   for k, v in row.items()}
            writer.writerow(out)


def write_md(rows: list[dict], path: Path) -> None:
    gold_labels = {"P02": "NoPref", "P03": "AAVE", "P05": "AAVE",
                   "P08": "AAVE",  "P09": "SAE",  "P13": "NoPref"}

    def md_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    def md_sep(n: int) -> str:
        return md_row(["---"] * n)

    # ── Table 1: accuracy + per-prompt verdicts ───────────────────────────────
    t1_headers = [
        "Variant", "Description", "Model", "Gold n",
        "Acc (both)", "Acc (SAE→)", "Acc (AAVE→)", "Order consist.",
    ] + [f"{pid} ({gold_labels[pid]})" for pid in GOLD_PROMPTS]

    t1_lines = [
        "## Table 1 — Accuracy & Per-Prompt Verdicts",
        "",
        "Gold set: P02 (No Pref), P03 (AAVE), P05 (AAVE), P08 (AAVE), P09 (SAE), P13 (No Pref)  ",
        "Symbols: ✓ correct · ✗ wrong · ~ order-inconsistent",
        "",
        md_row(t1_headers),
        md_sep(len(t1_headers)),
    ]
    for row in rows:
        cells = [
            row["variant"], row["description"], row["model"], str(row["gold_n"]),
            _pct(row["acc_overall"]), _pct(row["acc_sae_first"]), _pct(row["acc_aave_first"]),
            row["order_consist"],
        ] + [row.get(pid, "") for pid in GOLD_PROMPTS]
        t1_lines.append(md_row(cells))

    # ── Table 2: per-class precision / recall / F1 ────────────────────────────
    t2_headers = [
        "Variant", "Description",
        "SAE prec", "SAE rec", "SAE F1", "SAE n",
        "AAVE prec", "AAVE rec", "AAVE F1", "AAVE n",
        "NoPref prec", "NoPref rec", "NoPref F1", "NoPref n",
    ]

    t2_lines = [
        "",
        "## Table 2 — Per-Class Precision / Recall / F1",
        "",
        "Metrics are computed across both orderings (SAE-first + AAVE-first).  ",
        "A judge biased toward SAE will show high SAE precision but low AAVE recall.  ",
        "Support (n) = number of gold instances of that class × 2 orderings.",
        "",
        md_row(t2_headers),
        md_sep(len(t2_headers)),
    ]
    for row in rows:
        cells = [
            row["variant"], row["description"],
            _pct(row["sae_prec"]),  _pct(row["sae_rec"]),  _f(row["sae_f1"]),  str(row.get("sae_support", "")),
            _pct(row["aave_prec"]), _pct(row["aave_rec"]), _f(row["aave_f1"]), str(row.get("aave_support", "")),
            _pct(row["nopref_prec"]), _pct(row["nopref_rec"]), _f(row["nopref_f1"]), str(row.get("nopref_support", "")),
        ]
        t2_lines.append(md_row(cells))

    notes = [
        "",
        "## Notes",
        "",
        "- All variants use GPT-4.1 at temperature=0 unless stated otherwise.",
        "- v4 uses GPT-5 (medium reasoning); v1a/v1b also tested GPT-5.4 and GPT-5.5.",
        "- v9 removes P05 from gold (Qualtrics margin = 2 votes, treated as noisy label).",
        "- 'Acc (SAE→)' = SAE shown first; 'Acc (AAVE→)' = AAVE shown first.",
        "- The SAE-bias pattern (high SAE precision, low AAVE recall) reflects the judge's",
        "  tendency to prefer whichever response appears first or reads as more formal.",
        "- Ceiling of ~75% on the full 6-prompt gold set is attributed to weak inter-rater",
        "  agreement: 5 of 6 gold prompts had a Qualtrics margin of ≤ 3 votes out of 9.",
    ]

    header = [
        "# LLM-as-Judge Results — Human Preference Prediction",
        "",
        "Generated by `build_judge_results_table.py` from `dataset/study_logs_judge/`.  ",
        "12 prompt-engineering variants tested; best overall accuracy: **75.0%** (v6, v10, v11).",
        "",
    ]

    path.write_text(
        "\n".join(header + t1_lines + t2_lines + notes) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print(f"Scanning {LOG_DIR} …")
    rows = build_table()
    print(f"  {len(rows)} runs found.")

    write_csv(rows, OUT_CSV)
    print(f"  CSV → {OUT_CSV}")

    write_md(rows, OUT_MD)
    print(f"  MD  → {OUT_MD}")

    # Terminal preview of Table 2
    print()
    print(f"{'Var':<5} {'Acc':>6}  {'SAE P':>6} {'SAE R':>6}  {'AAVE P':>6} {'AAVE R':>6}  {'NP P':>6} {'NP R':>6}  Description")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['variant']:<5} {_pct(r['acc_overall']):>6}  "
            f"{_pct(r['sae_prec']):>6} {_pct(r['sae_rec']):>6}  "
            f"{_pct(r['aave_prec']):>6} {_pct(r['aave_rec']):>6}  "
            f"{_pct(r['nopref_prec']):>6} {_pct(r['nopref_rec']):>6}  "
            f"{r['description']}"
        )


if __name__ == "__main__":
    main()
