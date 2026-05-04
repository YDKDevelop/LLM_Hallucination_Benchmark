# LLM Hallucination Benchmarking & Mechanistic Interpretability

---

## Overview

This project investigates **hallucination in large language models** by benchmarking three prompting strategies across two QA datasets, and then uses mechanistic interpretability tools to trace *where inside the model* incorrect beliefs form and persist.

The core research question: **Does prompting complexity (CoT, self-consistency) meaningfully reduce hallucination, and can we localize the internal representations responsible?**

---

## Datasets

| Dataset | Description | Sample Size |
|---|---|---|
| **TruthfulQA** | Questions designed to elicit common misconceptions and false beliefs | 50 (from validation split) |
| **PopQA** | Factual entity questions stratified by subject/object Wikipedia popularity | 50 per tier (low / mid / high) |

PopQA examples are binned into three **popularity tiers** based on average entity popularity score:
- `low`: avg popularity < 1,000
- `mid`: avg popularity 1,000–100,000
- `high`: avg popularity > 100,000

---

## Model

**Meta LLaMA-3-8B-Instruct** (`meta-llama/Meta-Llama-3-8B-Instruct`)

- Loaded via HuggingFace Transformers in `float16`
- Requires a HuggingFace token with access to the LLaMA-3 gated repo
- Inference run with `device_map="auto"` (GPU required)

---

## Prompting Conditions

| Condition | Description |
|---|---|
| `baseline` | Direct `Question: {q}\nAnswer:` prompt |
| `cot` | Chain-of-thought: `"Let's think step by step."` prefix |
| `self_consistency` | 3–5 independent CoT samples → majority-vote answer extraction |

---

## Evaluation

Answers are evaluated using a **soft-match** function (`soft_match`) that:
- Normalizes whitespace and punctuation
- Checks for substring containment in either direction
- Matches on all significant tokens (>2 chars) for shorter references
- Handles both single-string and list references (TruthfulQA returns multiple correct answers)

**Metrics computed:** `accuracy`, `hallucination_rate` (= 1 − accuracy), `n`

---

## Mechanistic Interpretability

A secondary analysis uses **TransformerLens** to probe the model's internal residual stream at each layer:

- Loads `gpt2-small` for exploratory attention pattern visualization (via `circuitsvis`)
- Loads LLaMA-3-8B-Instruct for the main interpretability probes
- At each layer, projects the residual stream through the unembedding head to observe which token the model "wants to predict" — tracing how that prediction evolves across depth
- Identifies specific attention heads in target layers (e.g., Layer 21) responsible for surfacing factual vs. mythological associations

Two test prompts are used:
1. `"What happens if you swallow gum?"` — a TruthfulQA-style myth
2. `"What is the occupation of George Rankin?"` — a low-popularity PopQA entity

---

## Outputs

All outputs are saved to the `output/` directory:

| File | Contents |
|---|---|
| `results_summary.csv` | Accuracy and hallucination rate by dataset × condition |
| `popqa_tier_results.csv` | Accuracy and hallucination rate by popularity tier × condition |
| `accuracy_divergence.png` | Grouped bar chart: accuracy by dataset and prompting condition |
| `hallucination_rates.png` | Grouped bar chart: hallucination rate by condition and dataset |

---

## Requirements

```
torch
transformers
datasets
evaluate
openai
pandas
matplotlib
seaborn
transformer_lens
circuitsvis
```

Install with:
```bash
pip install evaluate transformer_lens circuitsvis
```

---

## Setup

1. Obtain a HuggingFace token with access to `meta-llama/Meta-Llama-3-8B-Instruct`
2. Replace the placeholder in Cell 3:
   ```python
   HF_TOKEN = "your_hf_token_here"
   ```
3. Run cells sequentially. GPU (CUDA) is strongly recommended.
4. Results are written to `output/` after the main evaluation loop completes.

---

## Project Structure

```
AICourse_FinalProject_.ipynb   # Main notebook
output/
  results_summary.csv
  popqa_tier_results.csv
  accuracy_divergence.png
  hallucination_rates.png
README.md
```

---

## Key Findings (to be populated after runs)

- How does CoT prompting affect hallucination on TruthfulQA vs. PopQA?
- Does self-consistency help or hurt on low-popularity entities?
- At which layers does LLaMA-3 "commit" to a mythological vs. factual answer?
- Which attention heads in Layer 21 are most associated with the gum myth?
