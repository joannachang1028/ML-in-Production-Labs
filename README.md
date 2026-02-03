# cmu-mlip-model-testing-lab

# Lab 4: Model Testing with Weights & Biases and LLMs

This lab walks you through evaluating a candidate sentiment model against a baseline by combining **Weights & Biases (W&B)** slice-based analysis with synthetic test generation from **Large Language Models (LLMs)**. You will log predictions, add failure-relevant metadata, compare models slice-by-slice, quantify regressions, and then use LLMs to stress test a weakness you uncovered.

## Prerequisites
- **Python 3.10+** (but W&B and the notebook layers work with Python ≥ 3.7)
- Git (the repo is already cloned via `git clone https://github.com/nikitachaudharicodes/cmu-mlip-model-testing-lab`)
- A free W&B account tied to your CMU email (see the W&B login section below)

## Deliverables checklist
- [ ] Run Steps 1–4, define at least five failure-relevant slices, and be ready to explain exactly why you chose each slice (what hypothesis it tests).
- [ ] Log the tables/metrics from Step 6 to W&B, explore the slices there, and capture your observations in the notebook so you can walk the TA through your charts and answer their questions.
- [ ] Finish the targeted stress test in Step 7, paste your hypothesis + 10 LLM tweets into the notebook, interpret what happened, and discuss that experiment with the TA before class.


## Install dependencies
```bash
pip install --upgrade wandb datasets transformers evaluate tqdm emoji regex pandas pyarrow scikit-learn nbformat torch
```

## W&B setup
1. Create a free W&B account at https://wandb.ai (use your CMU email if possible).
2. Copy your API key from https://wandb.ai/authorize.
3. Run `wandb login` in the terminal and paste the API key when prompted.

## Running the lab
1. Open `lab4.ipynb` in Jupyter/Colab/VS Code notebooks.
2. Run the notebook cells in order; there are seven numbered steps:
   1. **Setup** – install libraries, set random seed, and configure `PROJECT`/`ENTITY`/`RUN_NAME`.
   2. **Dataset** – load either `cardiffnlp/tweet_eval` (default) or the provided `tweets.csv` fallback; keep only `text` + `label`.
   3. **Metadata** – add at least five failure-relevant metadata columns (emoji count, hashtags, mentions, negations, length buckets, etc.) and declare helper slices such as negation/hashtag/emoji.
   4. **Inference** – run both the baseline and candidate models (Roberta vs. GPT-2) on all texts, then reshape the predictions into `df_long` and `df_wide`.
   5. **Metrics** – compute overall accuracy, slice accuracies, and regression-aware metrics (regressions, improvements, confident regressions).
   6. **W&B logging** – log `df_long`, `slice_metrics`, `regression_metrics`, and `df_eval` tables to W&B, plus model/regression summaries. Use the W&B run listed at the end to explore slices and charts.
   7. **LLM stress testing** – choose a weak slice, write a hypothesis, paste 10 LLM-generated tweets into `generated_cases`, run `run_on_generated_tests`, and (optionally) log back to W&B.

## Tips
- If `cardiffnlp/tweet_eval` is slow or blocked, set `USE_HF_DATASET = False` and the notebook will read from `tweets.csv`.
- Always run `wandb login` outside of the notebook to avoid exposing your API key in saved outputs.
- Keep the notebook outputs clean before turning the lab in; the README no longer references the deleted `images/` assets, so you can clear output and re-save.

## References
- W&B documentation on panels and tables: https://docs.wandb.ai/guides/app/features/panels/
