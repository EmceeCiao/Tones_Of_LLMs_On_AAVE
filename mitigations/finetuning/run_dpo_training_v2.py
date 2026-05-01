#!/usr/bin/env python3
"""
run_dpo_training_v2.py

Uploads openai_dpo_train_v2.jsonl (exact OpenAI DPO format with tools[] and
parallel_tool_calls) and launches a DPO fine-tuning job.

Usage:
    OPENAI_API_KEY=<key> python mitigations/finetuning/run_dpo_training_v2.py

    # To monitor an existing job without re-uploading:
    OPENAI_API_KEY=<key> python mitigations/finetuning/run_dpo_training_v2.py --job-id ftjob-abc123
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

SCRIPT_DIR    = Path(__file__).resolve().parent
TRAINING_FILE = SCRIPT_DIR / "openai_dpo_train_v2.jsonl"

BASE_MODEL = "gpt-4.1-2025-04-14"
DPO_BETA   = 0.1   # KL-penalty coefficient; lower = more deviation from base model allowed


def upload_training_file(client: OpenAI) -> str:
    print(f"Uploading {TRAINING_FILE.name} …")
    with TRAINING_FILE.open("rb") as fh:
        response = client.files.create(file=fh, purpose="fine-tune")
    print(f"  File uploaded → id: {response.id}  status: {response.status}")
    return response.id


def create_job(client: OpenAI, file_id: str) -> str:
    print(f"\nCreating DPO fine-tuning job on {BASE_MODEL} …")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL,
        method={
            "type": "dpo",
            "dpo": {
                "hyperparameters": {"beta": DPO_BETA},
            },
        },
    )
    print(f"  Job created → id: {job.id}  status: {job.status}")
    return job.id


def poll_job(client: OpenAI, job_id: str, poll_interval: int = 60) -> None:
    print(f"\nPolling job {job_id} every {poll_interval}s (Ctrl-C to stop) …\n")
    terminal_statuses = {"succeeded", "failed", "cancelled"}
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  [{time.strftime('%H:%M:%S')}] status: {job.status}", end="")
        if job.fine_tuned_model:
            print(f"  →  model: {job.fine_tuned_model}", end="")
        print()

        if job.status in terminal_statuses:
            if job.status == "succeeded":
                print(f"\nFine-tuned model id: {job.fine_tuned_model}")
            else:
                print(f"\nJob ended with status: {job.status}")
            break

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch or monitor an OpenAI DPO fine-tuning job.")
    parser.add_argument("--job-id", help="Skip upload/create and monitor an existing job id.")
    parser.add_argument("--no-poll", action="store_true", help="Exit after creating the job without polling.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    if args.job_id:
        job_id = args.job_id
        print(f"Monitoring existing job: {job_id}")
    else:
        file_id = upload_training_file(client)
        job_id  = create_job(client, file_id)

    if not args.no_poll:
        poll_job(client, job_id)


if __name__ == "__main__":
    main()
