# Resources
[OpenAPI DPO Article](https://developers.openai.com/cookbook/examples/fine_tuning_direct_preference_optimization_guide#5-fine-tuning)
# Workflow 
1) Gather synthetic data based on survey responses
    1) Generate prompt to generate synthetic data (optional)
    2) Use prompt to generate the synthetic data
    3) Benchmark with baseline model with LLM-as-a-judge
2) Combine the synthetic data with the leftover prompts and prompts from round 1 survey. Each training example should be formatted as the following: 
```
{
  "input": {
    "messages": [
      {
        "role": "user",
        "content": [prompt]
      }
    ],
    "tools": [],
    "parallel_tool_calls": true
  },
  "preferred_output": [
    {
      "role": "assistant",
      "content": [preferred response]
    }
  ],
  "non_preferred_output": [
    {
      "role": "assistant",
      "content": [non-preferred response]
    }
  ]
}
```
3) Finetune the data

### Prompt Generation - In More Detail
According to the doc, we can use random prompts from the survey response as a prompt seed pool. 
This is the following prompt that we can use to generate SAE & AAVE synthetic pairs from existing dataset.
```
Return TWO distinct, realistic customer-service question related in topic or theme to the following question but NOT a direct paraphrase. One should be SAE and another should be in AAVE. 
For example: 
SAE: What geometric shape is used in equations to determine net force?
AAVE: What shape do they use in them equations to figure out net force?
SAE: For most organisms, what is the dominant system of defense?
AAVE: For most living things, what's the main defense system?
SAE: Write a function to sort the given list.\nassert python_function([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])==[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\nGenerate a Python function to solve this problem. Ensure the generated function is named as python_function.
AAVE: Cook up a function that be sortin' the given list.\nassert python_function([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])==[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\nYou gotta whip up a Python function to handle this problem. You gon' make sure the function name right, which gotta python_function.
SAE: When I delete a picture or program, what actually happens to it?
AAVE: What actually happen when I delete a pic or app?

Question: [Insert Question]
```
However, since we have 1000 SQuAD_AAVE prompts from Will in the other paper, we will use that in addition to round 1 prompts + leftover prompts as the SAE/AAVE prompt pairs as the AAVE prompt is verfied by AAVE speakers. Deduplication need to be done, which can be easily done by searching the `xlsx_row` field in  `dataset/SQuAD_AAVE_Will/ablated.json` file.

## LLM-as-a-judge - In More Detail 
For each SAE/AAVE prompt pair, we will prompt the LLM for SAE/AAVE responses. 

To generate the synthetic preference data, we will do a few-shot prompt for the LLM with provided guidelines in response preferences. There are 2 options we have considered for how we do few shot prompting: 
1) Give SAE/AAVE prompt/response pairs + majority vote on response preferred
2) Give SAE/AAVE prompt/response pairs + average likert scale answers (warmth/clarity/helpfulness) + majority vote on response preferred

These few-shot examples are from the survey responses - we can choose to provide 10 round 1 prompts for few-shot prompting and the remaining 5 prompts as the held-out set for evaluation for how good is LLM-as-the-judge technique. 

In addition, we can use 3 LLMs to judge vote for the preferred responses, given the rubric (which is the few shot prompt). If all 3 LLMs agree on a particular preferred response, then the training example pair should be added to the finetuning data. Otherwise, if it's majority vote, average confidence score must be high for the training example pair to be added. In other cases, we drop the training example pair. 

## Amount of Finetuning Data 
```
Round 1 Prompts: 15 
Leftover Prompts: 57 + 9 Round 2 Prompts = 66
Will's SQuAD prompts: 1000 - 35 = 965

Total prompts: 1046 Prompts

Total Training Examples for Preference Data: 1046*2 = 2092
```

## DPO Finetuning Data Processing - In More Detail
We are going to decide on the preference label for the human responses with the following criteria: 
1) If a response is preferred, put the preference label for that preferred response
2) Otherwise, choose the response that has the highest average likert scale scores
3) If the scores are tied, drop or randomize the response to be chosen

For LLM-as-a-judge, if the response preferred is neither, then we drop the training example pair of those responses. 
