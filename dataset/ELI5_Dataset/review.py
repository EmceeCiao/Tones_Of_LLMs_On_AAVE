import json
import os

BATCH_JSON = "batch_to_review.json"
REVIEWED_JSONL = "reviewed.jsonl"

existing_ids = set()

if os.path.exists(REVIEWED_JSONL):
    with open(REVIEWED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            existing_ids.add(str(row["id"]))

with open(BATCH_JSON, "r", encoding="utf-8") as f:
    batch = json.load(f)

to_add = []
for row in batch:
    row_id = str(row["id"])
    if row_id not in existing_ids:
        to_add.append(row)
        existing_ids.add(row_id)

with open(REVIEWED_JSONL, "a", encoding="utf-8") as f:
    for row in to_add:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Added {len(to_add)} items to {REVIEWED_JSONL}")