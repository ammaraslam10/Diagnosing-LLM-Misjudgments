from __future__ import annotations

import json
import os
import time
from pathlib import Path
from radon.raw import analyze
from openai import OpenAI
from radon.complexity import cc_rank, cc_visit
from radon.metrics import h_visit, mi_visit

IN_DIR = Path("extracted_codes")
OUT_DIR = Path("extracted_codes_fixed")
LOG_PATH = Path("repair_log.jsonl")

MODEL = "gpt-4o"


SYSTEM_PROMPT = """You repair Python source code so that static analysis tools can run.

Rules:
- Make the smallest possible changes.
- Preserve original intent and identifiers.
- Do NOT refactor or reformat unless required to fix syntax/indentation/encoding.
- Fix indentation errors (missing blocks after ':' by inserting an indented 'pass').
- Replace invalid fullwidth punctuation with ASCII equivalents only when needed.
- If code contains stray non-Python text, comment it out rather than deleting.
- Output ONLY the repaired Python code. No markdown, no explanation.
"""


def llm_repair(code: str, err: str, filename: str, client: OpenAI, max_retries: int = 3) -> str:
    prompt = f"""File: {filename}

The tool failed with this error:
{err}

Repair the code following the rules.
CODE:
{code}
"""

    # simple retry with backoff
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            fixed = (resp.output_text or "").strip()
            if not fixed:
                raise RuntimeError("Empty LLM output")
            return fixed
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5 * (2 ** attempt))


def safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def main() -> None:
    if not IN_DIR.exists():
        raise SystemExit(f"Input folder not found: {IN_DIR.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key="YOUR_API_KEY_HERE")

    total = 0
    repaired = 0
    unchanged = 0
    failed = 0

    for src_path in IN_DIR.rglob("*.py"):
        total += 1
        rel = src_path.relative_to(IN_DIR)
        dst_path = OUT_DIR / rel

        code = src_path.read_text(encoding="utf-8", errors="replace")

        # 1) Try radon.raw.analyze
        try:
            analyze(code)  # if it doesn't raise, we consider it "ok"
            cc_visit(code)
            mi_visit(code, True)
            h_visit(code)
            safe_write_text(dst_path, code)
            unchanged += 1
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "file": str(rel),
                    "status": "ok",
                }) + "\n")
            continue
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        # 2) LLM repair
        try:
            fixed = llm_repair(code, err, str(rel), client=client)

            # remove ```python ... ``` if present
            if fixed.startswith("```") and fixed.endswith("```"):
                fixed = "\n".join(fixed.splitlines()[1:-1]).strip()

            # 3) Validate by running analyze again
            analyze(fixed)
            cc_visit(fixed)
            mi_visit(fixed, True)
            h_visit(fixed)
            safe_write_text(dst_path, fixed)
            repaired += 1
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "file": str(rel),
                    "status": "repaired",
                    "original_error": err,
                }) + "\n")
        except Exception as e2:
            print(f"Failed to repair {rel}: {type(e2).__name__}: {e2}")
            failed += 1
            # write original (or skip). Here we write original so every file exists in output.
            safe_write_text(dst_path, code)
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "file": str(rel),
                    "status": "failed",
                    "original_error": err,
                    "repair_error": f"{type(e2).__name__}: {e2}",
                }) + "\n")

    print(f"Done. total={total} ok={unchanged} repaired={repaired} failed={failed}")
    print(f"Output folder: {OUT_DIR.resolve()}")
    print(f"Log: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()

# total=1860 ok=1694 repaired=156 failed=10