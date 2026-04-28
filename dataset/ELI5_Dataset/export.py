import json

REVIEWED_JSONL = "reviewed.jsonl"
KEPT_JSON = "kept_questions.json"
MAYBE_JSON = "maybe_questions.json"

kept = []
maybe = []

with open(REVIEWED_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        status = row.get("status", "").strip().lower()

        if status == "keep":
            kept.append(row)
        elif status == "maybe":
            maybe.append(row)

with open(KEPT_JSON, "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=2, ensure_ascii=False)

with open(MAYBE_JSON, "w", encoding="utf-8") as f:
    json.dump(maybe, f, indent=2, ensure_ascii=False)

print(f"Wrote {len(kept)} kept questions to {KEPT_JSON}")
print(f"Wrote {len(maybe)} maybe questions to {MAYBE_JSON}")