# Tones of LLMs on AAVE: How Dialect Shapes GPT-4.1 Responses

This repository contains the prompt curation pipeline, response generation scripts, human evaluation processing, mitigation experiments, similarity analysis, and figure/table generation code for our study of how LLM responses differ when semantically equivalent prompts are written in Standard American English (SAE) versus African American Vernacular English (AAVE) for GPT-4.1

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Scripts that call the OpenAI API require:

```bash
export OPENAI_API_KEY="your-key-here"
```

Analysis, figure, and table scripts that read from saved `.xlsx` or `.jsonl` outputs do not require any API calls.

## Prompt Format

The pipeline starts from curated SAE/AAVE prompt pairs stored in [`Final_Dataset/final_prompts.json`](Final_Dataset/final_prompts.json):

```json
{
  "round 1": [
    {
      "category": "QA",
      "sae_prompt": "What project put the first Americans into space?",
      "aave_prompt": "What project got the first Americans up in space?"
    }
  ],
  "round 2": [],
  "leftover": []
}
```

Each entry requires:

- `category`: task type — `Algorithm`, `QA`, `ELI5`, or `Social`
- `sae_prompt`: Standard American English version of the prompt
- `aave_prompt`: AAVE translation of the same underlying question

## Response Generation

Generate responses for each SAE/AAVE prompt pair using the OpenAI Responses API:

```bash
python scripts/generate_responses_from_final_prompts.py
```

Check the constants at the top of the script before running (`INPUT_JSON_PATH`, `ROUND_SCOPE`, `MODEL_NAME`, `N_TRIALS`, `OUTPUT_DIR`). The saved Round 1 response dataset used by all downstream scripts is:

```text
Final_Dataset/final_prompt_responses.json
```

Round 2 uses the selected mitigation (fine-tuned model + system prompt):

```bash
python scripts/generate_responses_round2.py
```

## Qualtrics Survey Construction

Convert response logs into a Qualtrics-ready JSON list:

```bash
python scripts/build_qualtrics_json.py \
  Final_Dataset/study_logs/round1_20260419T154412Z.jsonl \
  --round 1 \
  --output Final_Dataset/Qualtrics/Round_1/Round_1_Qualtrics.json
```

Render into HTML-ready survey assets:

```bash
python scripts/qualtrics_html_converter_v5.py \
  Final_Dataset/Qualtrics/Round_1/Round_1_Qualtrics.json \
  --fields prompt response_A response_B \
  --output-json Final_Dataset/Qualtrics/Round_1/Round_1_Qualtrics_html.json \
  --preview-html Final_Dataset/Qualtrics/Round_1/Round_1_Qualtrics_preview.html \
  --snippet-dir Final_Dataset/Qualtrics/Round_1/Round_1_snippets
```

The Qualtrics Surveys sent out can be found in the root of the repository and are labeled Round_1.qsf & Round_2.qsf as this is the format supported by Qualtrics for importing and exporting surveys. 

## Qualtrics Aggregation

After collecting survey responses, normalize the randomized A/B columns so ratings map back to the correct dialect:

```bash
# Round 1
python Round_1_Results/Qualtrics_Data/aggregate_round1_qualtrics.py
python Round_1_Results/Qualtrics_Data/round1_summary.py

# Round 2
python Round_2_Results/Qualtrics_Data/aggregate_round2_qualtrics.py
python Round_2_Results/Qualtrics_Data/round2_summary.py
```

Outputs:

```text
Round_1_Results/Qualtrics_Data/*_aggregated.xlsx
Round_1_Results/Qualtrics_Data/*_aggregated_simple_summary.xlsx
Round_2_Results/Qualtrics_Data/*_aggregated.xlsx
Round_2_Results/Qualtrics_Data/*_aggregated_simple_summary.xlsx
```

Each summary workbook contains prompt-level and category-level sheets with helpfulness, clarity, warmth, tone composite, preference vote counts, preference majority, and tone winner.

## Similarity Analysis

Compute cosine similarity (`text-embedding-3-large`), token Jaccard, and bigram Jaccard between each SAE/AAVE response pair:

```bash
python scripts/round1_similarity.py
python scripts/round2_similarity.py
```

Outputs:

```text
Final_Dataset/round1_similarity.xlsx
Final_Dataset/round2_similarity.xlsx
```

## Repeated-Run Controls

To separate dialect effects from stochastic temperature variation, five responses are generated per prompt per dialect and compared within and across dialects:

| Comparison | Description |
| --- | --- |
| **Within SAE** | Similarity among the 5 SAE responses for the same prompt |
| **Within AAVE** | Similarity among the 5 AAVE responses for the same prompt |
| **Between matched** | SAE trial *k* vs AAVE trial *k* |
| **Between all pairs** | Every SAE trial vs every AAVE trial |

Collect repeated-run responses:

```bash
python scripts/collect_round1_repeated_runs.py
python scripts/collect_round2_repeated_runs.py
```

Analyze and run permutation tests:

```bash
python scripts/round1_repeated_runs_similarity.py
python scripts/round2_repeated_runs_similarity.py
```

Outputs:

```text
Final_Dataset/round1_repeated_runs_similarity.xlsx
Final_Dataset/round2_repeated_runs_similarity.xlsx
```

## Mitigation and Fine-Tuning

Round 1 human preferences and user-attribution judge outputs are combined into DPO training pairs:

```bash
python mitigations/finetuning/build_dpo_from_qualtrics.py
python mitigations/finetuning/build_openai_dpo.py
python mitigations/finetuning/run_dpo_training_v2.py
```

Key files:

```text
mitigations/system_prompt.txt
mitigations/finetuning/qualtrics_dpo_pairs.jsonl
mitigations/finetuning/user_attribution_dpo_pairs.jsonl
mitigations/finetuning/openai_dpo_train.jsonl
```

Test the mitigated conditions against the original Round 1 prompts:

```bash
python mitigations/comparison/mitigation_check.py
python mitigations/comparison/mitigation_run_judge.py
```

Outputs:

```text
mitigations/comparison/mitigation_study_logs/*_verdicts.jsonl
mitigations/comparison/mitigation_study_logs/judge_comparison_summary.txt
```

Scripts in `mitigations/Failed_Attempts_At_LLM_As_Judge/` are archived experiments and are not part of the final pipeline.

## Figures

Generate all main publication figures:

```bash
python scripts/figures.py
python scripts/permutation_test_figure.py
```

Generate Qualtrics/human-evaluation figures:

```bash
python qualtrics_figures/round1/qualtrics_figures_round1.py
python qualtrics_figures/round2/qualtrics_figures_round2.py
```

All outputs are written to:

```text
figures/
qualtrics_figures/round1/figures/
qualtrics_figures/round2/figures/
```

## LaTeX Tables

Regenerate all LaTeX tables from saved `.xlsx` outputs without rerunning API calls:

```bash
python latex_tables.py
```

Individual tables can also be compiled directly from `latex_tables/`. Add the following to your LaTeX preamble:

```latex
\usepackage{booktabs}
\usepackage{caption}
\usepackage{pdflscape}
```

## Repository Layout

```text
Final_Dataset/
  final_prompts.json                         # SAE/AAVE prompt pairs
  final_prompt_responses.json                # Round 1 prompts + generated responses
  round1_similarity.xlsx                     # Round 1 SAE/AAVE similarity results
  round2_similarity.xlsx                     # Round 2 SAE/AAVE similarity results
  round1_repeated_runs_similarity.xlsx       # Round 1 repeated-run control results
  round2_repeated_runs_similarity.xlsx       # Round 2 repeated-run control results
  study_logs/                                # JSONL generation logs
  Qualtrics/                                 # Qualtrics survey JSON/HTML/snippets

Round_1_Results/Qualtrics_Data/             # Round 1 raw + aggregated Qualtrics data
Round_2_Results/Qualtrics_Data/             # Round 2 raw + aggregated Qualtrics data

scripts/                                     # Generation, similarity, figure scripts
mitigations/                                 # DPO pairs, fine-tuning, judge comparison
qualtrics_figures/                           # Human-eval figures (round1/, round2/)
figures/                                     # Main publication figures
latex_tables/                                # Generated LaTeX table files
```