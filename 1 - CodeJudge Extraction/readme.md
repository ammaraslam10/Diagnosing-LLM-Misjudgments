# APPS dataset evaluator

These are some scripts to evaluate APPS dataset locally. Please download json files from https://huggingface.co/datasets/CodeResearch/CodeJudge-Eval/tree/main and APPS dataset from https://github.com/hendrycks/apps?tab=readme-ov-file

These need to be in the same main directory.

The steps are as follows:

1. Run the script to add locations of APPS corresponding locations to the CodeJudge Eval dataset. This will automatically look for `CodeJudge_Eval_0shot_easy.json`, `CodeJudge_Eval_0shot_middle.json` and `CodeJudge_Eval_0shot_json.json` and APPS folder (both in corrent directory) and create `[filename]_with_locations.json`
```
python3 add_location_to_evals.py
```
2. Run evaluate_code_solutions script. The filename inside the script needs to be updated so that it picks correct `X_with_location.json` file. This script does multi-threading to first of all run the original APPS solution on its own code and then run all of the test cases on "code" field from `X_with_location.json`. Label and evaluation details are added and a. new `X_with_locations_with_evaluations.json` file is created. Some sols will be marked as "NA"
```
python3 evaluate_code_solutions.py
```

NA's occur in the following conditions:

1. The gold solution in APPS dataset failed on its own test cases
2. The `code` in the `X_with_location.json` file does not have any input statement and the arguments are not simple to pass (it doesn't parse I/O so the test cases could not be passed to the code, however the solution might be correct... it is in an unknowable state for the automatic system)