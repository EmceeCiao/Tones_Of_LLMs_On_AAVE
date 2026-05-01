import json
from pathlib import Path  
OUTPUT_FILE = "to_dpo_results.json"
INPUT_FILE = "./dataset/study_logs_dpo/dpo_responses_20260428T065048Z.jsonl" 

def extract_selected_fields(input_file, output_file=OUTPUT_FILE):
    text = Path(input_file).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()

    idx = 0
    n = len(text)
    results = []

    while idx < n:
        # Skip whitespace between JSON objects
        while idx < n and text[idx].isspace():
            idx += 1

        if idx >= n:
            break

        try:
            obj, next_idx = decoder.raw_decode(text, idx)
            idx = next_idx
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON near character {idx}: {e}")
            break

        if isinstance(obj, dict) and obj.get("record_type") == "trial_result":
            results.append({
                "prompt_index": obj.get("prompt_index"),
                "category": obj.get("category"),
                "round": obj.get("round"),
                "prompt_text": obj.get("prompt_text"),
                "dialect": obj.get("dialect"),
                "trial": obj.get("trial"),
                "response_text": obj.get("response_text"), 
                "reasoning_text": obj.get("reasoning_text")
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} records to {output_file}")


if __name__ == "__main__":
    extract_selected_fields(INPUT_FILE)