# LLM Augmented Judgment

This folder contains code for performing judgment on evaluated code samples using an LLM. The workflow assumes that code evaluation and quality metrics have already been generated in previous steps.

## Overview

The scripts here run LLM-based judgment on the prepared evaluation data. Judgment can be performed with or without explicit reasoning prompts, depending on the desired output detail.

## Usage

Run the standard judgment script:

```bash
python3 ai_evaluate.py
```

Run the judgment script with reasoning prompts enabled:

```bash
python3 ai_evaluate_with_reasoning.py
```

## Files

- `ai_evaluate.py` - Runs LLM judgment on prepared results without additional reasoning prompts.
- `ai_evaluate_with_reasoning.py` - Runs LLM judgment with reasoning prompts included to produce more detailed explanations.

## Notes

- Ensure that the previous evaluation steps have completed and that the required input files are available.
- Verify your Python environment and dependencies before running the scripts.

