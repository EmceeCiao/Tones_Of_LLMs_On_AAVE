import json
import random
import os

SOURCE_JSONL = "eli5.jsonl"
REVIEWED_JSONL = "reviewed.jsonl"
BATCH_JSON = "batch_to_review.json"

BATCH_SIZE = 40
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

reviewed_ids = set()

if os.path.exists(REVIEWED_JSONL):
    with open(REVIEWED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            reviewed_ids.add(str(row["id"]))

candidates = []
with open(SOURCE_JSONL, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        row = json.loads(line)
        q = row.get("question", "").strip()
        if not q:
            continue

        row_id = str(row.get("id", i))
        if row_id in reviewed_ids:
            continue

        candidates.append({
            "id": row_id,
            "question": q,
            "status": "",
            "notes": ""
        })

sample_size = min(BATCH_SIZE, len(candidates))
batch = random.sample(candidates, sample_size)

with open(BATCH_JSON, "w", encoding="utf-8") as f:
    json.dump(batch, f, indent=2, ensure_ascii=False)

print(f"Wrote {sample_size} questions to {BATCH_JSON}")
print(f"{len(candidates) - sample_size} unseen questions remain.")