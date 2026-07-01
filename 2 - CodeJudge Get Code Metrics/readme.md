# Analysis for Code Eval with quality metrics

The pre-requisites is to have CodeJudge_Eval_0shot_X_with_locations_with_evaluation.json files ready from step 1.

1. Install requirements.txt
2. Run `extract_funcs.py`, it will create a folder called `extracted_codes` with individual files
```
python3 extract_funcs.py --input-dir [PATH] --output-dir [PATH]
```
3. This step attempts to fix syntax errors minimally (i.e. indentation issues etc) so that static analysis is possible for some tools i.e. radon & complexipy (ast is needed for these to work)
```
python3 fix_codes.py --input-dir [PATH] --output-dir [PATH] --api-key [OpenAI Key]
```
4. Run `metrics.py`, it will create a `CSV_reports` folder and generate CSVs with metrics (input-dir should contain extracted_codes and extracted_codes_fixed folders)
```
python3 metrics.py --input-dir [PATH] --output-dir [PATH]
```