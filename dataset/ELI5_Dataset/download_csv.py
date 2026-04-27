from datasets import load_dataset

ds = load_dataset("sentence-transformers/eli5", "pair", split="train")

# Save as CSV
ds.to_csv("eli5.csv")

# Save as JSONL
ds.to_json("eli5.jsonl") 