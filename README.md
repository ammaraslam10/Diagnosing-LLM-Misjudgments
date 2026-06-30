# CodeJudgeEval-X: Diagnosing LLM Misjudgments in Automated Code Evaluation

This repository contains the scripts used to build and analyze a misjudgement-focused version of the CodeJudge evaluation pipeline. The workflow augments the CodeJudge-Eval dataset with static code quality metrics and problem-level metrics, and includes the implementation of the Logistic Regression and Random Forest classifiers used in the study.

---

# Repository Structure

```text
Diagnosing-LLM-Misjudgments/
├── 1 - CodeJudge Extraction/
├── 2 - CodeJudge Get Code Metrics/
├── 3 - LLM Augmented Judgement/
├── 4 - Analysis/
├── Dataset/
├── Outputs/
├── Dockerfile
├── requirements.txt
├── README.md
└── citation.cff
```

---

# Reproducibility

This repository supports two methods for reproducing the experiments:


1. **Local Python Environment**: Install the required Python packages manually.
2. **Using Docker (Recommended)**: Reproduces the exact software environment used in our experiments.

---


# Option 1: Local Installation

Clone the repository:

```bash
git clone https://github.com/ammaraslam10/Diagnosing-LLM-Misjudgments.git
cd Diagnosing-LLM-Misjudgments
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the same scripts listed above directly from your local machine.

---

# Pipeline Overview

## Step 1. Code Extraction, Gold Label Creation, and Problem-Level Metrics

Builds the base evaluation files.

Main scripts:

* `add_location_to_evals.py` attaches APPS problem locations to CodeJudge examples.
* `evaluate_code_solutions.py` executes generated code against the official test cases and stores execution-based labels.
* `add_additional_metrics_to_evals.py` enriches each example with additional metrics such as:

  * Problem difficulty
  * Prompt length
  * Solution length
  * Readability metrics
  * Prompt perplexity
  * API-call counts

---

## Step 2. Static Code Metrics

Extracts code snippets and computes static software quality metrics.

Main scripts:

* `extract_funcs.py`
* `fix_codes.py`
* `metrics.py`

The following tools are used:

* Radon
* Pylint
* Bandit
* Complexipy

The computed metrics are exported as CSV files.

---

## Step 3. LLM-Augmented Judgement

Uses an LLM to re-evaluate labelled examples.

Main scripts:

* `ai_evaluate.py`
* `ai_evaluate_with_reason.py`

These scripts require an OpenAI-compatible API key.

Input and output filenames are currently configured directly within the source code.

---

## Step 4. Analysis

Builds machine learning models to study which features are associated with LLM misjudgements.

Main scripts:

* `logistic_regression_misjudgement_classifier.py`
* `random_forest_misjudgement_classifier.py`

These scripts merge judgement data with the extracted code metrics, train classifiers, and generate reports and visualizations.

---
# Option 2: Reproduce Using Docker (Recommended)

The Docker image contains all required dependencies specified in `requirements.txt`.

## Build the Docker image

From the repository root, run:

```bash
docker build -t diagnosing_llm_misjudgements .
```

## Run the Docker container

```bash
docker run --rm -it diagnosing_llm_misjudgements
```

The repository will be available inside the container in the Docker working directory (for example, `/workspace` or `/app`, depending on the Dockerfile).

Run the required scripts for each stage of the pipeline.

### Step 1. Code Extraction, Gold Label Creation, and Problem-Level Metrics

```bash
cd "/workspace/1 - CodeJudge Extraction"

python add_location_to_evals.py
python evaluate_code_solutions.py
python add_additional_metrics_to_evals.py
```

### Step 2. Static Code Metrics

```bash
cd "/workspace/2 - CodeJudge Get Code Metrics"

python extract_funcs.py
python fix_codes.py
python metrics.py
```

### Step 3. LLM-Augmented Judgement

```bash
cd "/workspace/3 - LLM Augmented Judgement"

python ai_evaluate.py
python ai_evaluate_with_reason.py
```

### Step 4. Analysis

```bash
cd "/workspace/4 - Analysis"

python logistic_regression_misjudgement_classifier.py
python random_forest_misjudgement_classifier.py
```

> **Note:** If your Dockerfile uses `WORKDIR /app` instead of `/workspace`, simply replace `/workspace` with `/app` in the commands above.

---
# Outputs

Analysis results are written to the `Outputs/` directory.

```
Outputs/
├── easy/
├── middle/
└── hard/
```

Each difficulty folder contains:

* `logistic_regression_misjudgement_classifier.txt`
* `random_forest_misjudgement_classifier.txt`
* `report/`

The report directory includes artifacts such as:

* SHAP plots
* ROC curves
* Confusion matrices
* Feature importance plots
* CSV summaries

Additional beeswarm analysis is provided in:

```
beeswarm analysis.md
```

---

# Dataset

The processed datasets are located in the `Dataset/` directory.

Available files include:

* `CodeJudge_Eval_X_0shot_easy.json`
* `CodeJudge_Eval_X_0shot_middle.json`
* `CodeJudge_Eval_X_0shot_hard.json`
* `CodeJudge_Eval_reasoning_0shot_easy.json`

The first three files contain the CodeJudge-Eval zero-shot dataset augmented with code-level and problem-level features.

The final dataset contains LLM judgements together with the generated reasoning used for RQ4.

---



# Citation

Citation metadata for this repository is provided in `citation.cff`.
