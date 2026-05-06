#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FINAL_PROMPTS_PATH = SCRIPT_DIR / "Final_Dataset" / "final_prompts.json"
DEFAULT_QA_XLSX_PATH = SCRIPT_DIR / "SQuAD_AAVE_Will" / "QA_translated_dataset.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "study_logs_dpo"

MODEL_NAME = "gpt-4.1"
N_TRIALS = 1
TEMPERATURE = 1
TOP_P = 1
REASONING_TYPE = "medium"

FINAL_ROUND_SCOPE = "leftover"

SAE_QA_HEADER = "Standard American English Question"
AAVE_QA_HEADER = "African American Vernacular English Equivalent (AAVE)"

XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def normalize_for_dedupe(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def qa_pair_key(sae_prompt: str | None, aave_prompt: str | None) -> tuple[str, str]:
    return (normalize_for_dedupe(sae_prompt), normalize_for_dedupe(aave_prompt))


def prompt_pair_key(item: dict) -> tuple[str, str, str]:
    return (
        normalize_for_dedupe(item.get("category")),
        normalize_for_dedupe(item.get("sae_prompt")),
        normalize_for_dedupe(item.get("aave_prompt")),
    )


def load_leftover_final_items(final_prompts_path: Path) -> tuple[list[dict], int]:
    with final_prompts_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected final prompts file to be grouped by round.")

    prompt_items = []
    seen_prompt_keys = set()
    skipped_duplicates = 0
    for index, item in enumerate(data.get(FINAL_ROUND_SCOPE, []), start=1):
        key = prompt_pair_key(item)
        if key in seen_prompt_keys:
            skipped_duplicates += 1
            continue
        seen_prompt_keys.add(key)

        prompt_items.append(
            {
                **item,
                "round": None,
                "source_dataset": "final_prompts",
                "source_split": FINAL_ROUND_SCOPE,
                "source_index": index,
            }
        )
    return prompt_items, skipped_duplicates


def column_letters(cell_ref: str) -> str:
    return re.sub(r"[^A-Z]", "", cell_ref.upper())


def read_shared_strings(archive: ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(si.itertext()) for si in root.findall("m:si", XLSX_NS)]


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find("m:v", XLSX_NS)
    if value_node is None or value_node.text is None:
        inline = cell.find("m:is", XLSX_NS)
        return "".join(inline.itertext()) if inline is not None else ""

    value = value_node.text
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value)]
    return value


def load_xlsx_rows(xlsx_path: Path) -> list[dict]:
    with ZipFile(xlsx_path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

        rows = []
        header_by_column = {}
        for row in sheet.findall(".//m:sheetData/m:row", XLSX_NS):
            row_number = int(row.attrib["r"])
            values_by_column = {
                column_letters(cell.attrib["r"]): cell_value(cell, shared_strings)
                for cell in row.findall("m:c", XLSX_NS)
            }

            if row_number == 1:
                header_by_column = values_by_column
                continue

            rows.append(
                {
                    header_by_column.get(column, column): value
                    for column, value in values_by_column.items()
                }
                | {"xlsx_row": row_number}
            )

    return rows


def load_qa_translated_items(
    qa_xlsx_path: Path, existing_final_qa_keys: set[tuple[str, str]]
) -> tuple[list[dict], int]:
    qa_items = []
    seen_keys = set(existing_final_qa_keys)
    skipped_duplicates = 0

    for row in load_xlsx_rows(qa_xlsx_path):
        sae_prompt = row.get(SAE_QA_HEADER)
        aave_prompt = row.get(AAVE_QA_HEADER)
        if not sae_prompt or not aave_prompt:
            continue

        key = qa_pair_key(sae_prompt, aave_prompt)
        if key in seen_keys:
            skipped_duplicates += 1
            continue

        seen_keys.add(key)
        qa_items.append(
            {
                "category": "QA",
                "sae_prompt": sae_prompt,
                "aave_prompt": aave_prompt,
                "round": None,
                "source_dataset": "QA_translated_dataset",
                "source_split": "all",
                "source_index": row["xlsx_row"],
                "xlsx_row": row["xlsx_row"],
            }
        )

    return qa_items, skipped_duplicates


def build_prompt_items(final_prompts_path: Path, qa_xlsx_path: Path) -> tuple[list[dict], dict]:
    final_items, skipped_final_duplicates = load_leftover_final_items(final_prompts_path)
    final_qa_keys = {
        qa_pair_key(item.get("sae_prompt"), item.get("aave_prompt"))
        for item in final_items
        if item.get("category") == "QA"
    }

    qa_items, skipped_qa_duplicates = load_qa_translated_items(qa_xlsx_path, final_qa_keys)
    prompt_items = final_items + qa_items

    counts = {
        "leftover_final_items_added": len(final_items),
        "leftover_final_duplicates_skipped": skipped_final_duplicates,
        "leftover_final_qa_items": len(final_qa_keys),
        "qa_translated_items_added": len(qa_items),
        "qa_translated_duplicates_skipped": skipped_qa_duplicates,
        "total_prompt_items": len(prompt_items),
    }
    return prompt_items, counts


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def collect_reasoning_text(response) -> str | None:
    reasoning_texts = []
    for output_item in getattr(response, "output", []) or []:
        if getattr(output_item, "type", None) == "reasoning":
            for summary in getattr(output_item, "summary", []) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning_texts.append(text)
    return "\n".join(reasoning_texts) if reasoning_texts else None


def create_response(client, prompt_text: str):
    return client.responses.create(
        model=MODEL_NAME,
        input=prompt_text,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one GPT-4.1 response per SAE/AAVE prompt for leftover final prompts "
            "and non-duplicate QA_translated_dataset rows."
        )
    )
    parser.add_argument("--final-prompts-path", type=Path, default=DEFAULT_FINAL_PROMPTS_PATH)
    parser.add_argument("--qa-xlsx-path", type=Path, default=DEFAULT_QA_XLSX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and de-duplicate inputs, print counts, and exit without calling OpenAI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_items, counts = build_prompt_items(args.final_prompts_path, args.qa_xlsx_path)

    if args.dry_run:
        print(json.dumps(counts, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = args.output_dir / f"dpo_responses_{run_id}.jsonl"

    experiment_config = {
        "run_id": run_id,
        "model": MODEL_NAME,
        "final_prompts_path": str(args.final_prompts_path),
        "final_round_scope": FINAL_ROUND_SCOPE,
        "qa_xlsx_path": str(args.qa_xlsx_path),
        "counts": counts,
        "n_trials_per_prompt_variant": N_TRIALS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    append_jsonl(
        log_path,
        {
            "record_type": "experiment_config",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": experiment_config,
        },
    )

    from openai import OpenAI

    client = OpenAI()

    for prompt_index, item in enumerate(prompt_items, start=1):
        category = item.get("category")
        round_num = item.get("round")

        for dialect_key, dialect_label in [("sae_prompt", "SAE"), ("aave_prompt", "AAVE")]:
            prompt_text = item.get(dialect_key)
            if not prompt_text:
                continue

            for trial in range(1, N_TRIALS + 1):
                trial_start = time.time()
                base_record = {
                    "record_type": "trial_result",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "config": experiment_config,
                    "prompt_index": prompt_index,
                    "category": category,
                    "round": round_num,
                    "source_dataset": item.get("source_dataset"),
                    "source_split": item.get("source_split"),
                    "source_index": item.get("source_index"),
                    "xlsx_row": item.get("xlsx_row"),
                    "dialect": dialect_label,
                    "prompt_text": prompt_text,
                    "trial": trial,
                    "reasoning_type": REASONING_TYPE,
                }

                try:
                    response = create_response(client, prompt_text)
                    append_jsonl(
                        log_path,
                        base_record
                        | {
                            "status": "success",
                            "response_id": getattr(response, "id", None),
                            "response_text": getattr(response, "output_text", None),
                            "system_fingerprint": getattr(response, "system_fingerprint", None),
                            "reasoning_text": collect_reasoning_text(response),
                            "seed_used_in_request": False,
                            "duration_sec": time.time() - trial_start,
                        },
                    )
                    print(f"Saved prompt {prompt_index} | {dialect_label} | trial {trial}")

                except Exception as e:
                    append_jsonl(
                        log_path,
                        base_record
                        | {
                            "status": "error",
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "duration_sec": time.time() - trial_start,
                        },
                    )
                    print(f"Prompt {prompt_index} | {dialect_label} | trial {trial} failed and was logged.")

    append_jsonl(
        log_path,
        {
            "record_type": "experiment_end",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "config": experiment_config,
        },
    )

    print(f"Done. Results saved to: {log_path}")


if __name__ == "__main__":
    main()
