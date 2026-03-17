# CodeJudgeEval-X: Diagnosing LLM Misjudgments in Automated

This repository contains the scripts used to build and analyze a misjudgement-focused version of the CodeJudge evaluation pipeline. The workflow augments CodeJudge-Eval dataset static code quality metrics and problem level metrics, and the code for the logistic & RF classifiers for the study.

## Dataset

The processed dataset used by this repository is available on Hugging Face:

- [CodeJudge-Eval-X](https://huggingface.co/datasets/ammaraslam10/CodeJudge-Eval-X)

## Repository structure

```text
Diagnosing-LLM-Misjudgments/
├── 1 - CodeJudge Extraction/
├── 2 - CodeJudge Get Code Metrics/
├── 3 - LLM Augmented Judgement/
├── 4 - Analysis/
└── requirements.txt
```

### Step 1. CodeJudge Extraction
Builds the base evaluation files.

Main scripts:
- `add_location_to_evals.py` attaches APPS problem locations to CodeJudge examples.
- `evaluate_code_solutions.py` runs generated code against tests and stores execution-based labels.
- `add_additional_metrics_to_evals.py` enriches each record with metrics such as difficulty, text lengths, readability measures, prompt perplexity, and API-call counts.

### Step 2. CodeJudge Get Code Metrics
Extracts code snippets and computes static quality metrics.

Main scripts:
- `extract_funcs.py` writes each solution into an individual Python file.
- `metrics.py` runs `radon`, `pylint`, `bandit`, and `complexipy`, then exports CSV reports.
- `fix_codes.py` helper for code repair for tools like radon/complexipy.

### Step 3. LLM Augmented Judgement
Uses an LLM to re-evaluate labelled examples.

Main scripts:
- `ai_evaluate.py` predicts judgement labels with an LLM and computes evaluation metrics.
- `ai_evaluate_with_reason.py` LLM needs to provide reasoning along with the output label.

These scripts expect an OpenAI-compatible API key and currently use file names configured directly in the source.

### Step 4. Analysis
Builds classifiers to study which features are associated with misjudgements.

Main scripts:
- `logistic_regression_misjudgement_classifier.py`
- `random_forest_misjudgement_classifier.py`

These scripts merge JSON judgement data with metric CSV files, train models, and generate reports and plots for feature importance and misjudgement prediction.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Citation

Citation metadata for the repository is provided in `citation.cff`.
