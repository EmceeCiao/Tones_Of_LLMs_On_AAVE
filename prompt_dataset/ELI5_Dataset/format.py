from pathlib import Path

input_file = Path("eli5.jsonl")
output_dir = Path("Formatted_Subsets")
lines_per_file = 2000

output_dir.mkdir(exist_ok=True)

with input_file.open("r", encoding="utf-8") as f:
    chunk_num = 1
    current_lines = []

    for i, line in enumerate(f, start=1):
        current_lines.append(line)
        if i % lines_per_file == 0:
            out_path = output_dir / f"eli5_part_{chunk_num:03d}.jsonl"
            with out_path.open("w", encoding="utf-8") as out:
                out.writelines(current_lines)
            current_lines = []
            chunk_num += 1

    if current_lines:
        out_path = output_dir / f"eli5_part_{chunk_num:03d}.jsonl"
        with out_path.open("w", encoding="utf-8") as out:
            out.writelines(current_lines)

print(f"Done. Wrote chunk files to: {output_dir}")