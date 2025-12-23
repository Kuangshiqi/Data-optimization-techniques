import json
import re
from typing import Dict, Any

from tqdm import tqdm
import requests

from test_mbpp import eval_code as eval_code_mbpp
from eval_code import eval_code as eval_code_apps_cc


def clean_generated_code(code: str) -> str:
    if not isinstance(code, str):
        return ""
    code = re.sub(r"^\s*```python\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def _detect_dataset(dataset: str) -> str:
    d = (dataset or "").lower().strip()
    if d in {"mbpp", "apps", "codecontests"}:
        return d
    raise ValueError("dataset must be one of: mbpp / apps / codecontests (or cc)")


def _get_prompt_and_tests(entry: Dict[str, Any], dataset: str):
    question = entry.get("question")
    in_outs = entry.get("input_output")
    if isinstance(in_outs, str):
        try:
            in_outs = json.loads(in_outs)
        except Exception:
            pass
    return question, in_outs


def generate_prompt(question: str) -> str:
    prompt = (
        "You will be given a question (problem specification) and will generate a correct Python program "
        "that matches the specification.\n\n"
    )
    prompt += f"Question: {question}\n\n"
    return prompt


def call_openrouter(prompt: str, api_key: str, model: str) -> str:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"{api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            }
        ),
        timeout=120,
    )
    response.raise_for_status()
    response_data = response.json()
    return response_data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


def _passes_tests(code: str, tests: Any, dataset: str) -> bool:
    if dataset == "mbpp":
        try:
            return float(eval_code_mbpp(code, tests)) == 1.0
        except Exception:
            return False

    try:
        return float(eval_code_apps_cc(tests, code)) == 1.0
    except Exception:
        return False


def synthesis(dataset: str) -> Dict[str, Any]:
    dataset = _detect_dataset(dataset)

    input_file = f"Datasets/{dataset}/train_ori.jsonl"
    output_file = f"Datasets/{dataset}/train_aug.jsonl"
    api_key = ""  # OpenRouter key
    api_model = ""  # e.g. "google/gemini-2.0-flash-001"
    max_attempts = 5

    total_attempts = 0
    failed_attempts = 0

    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(output_file, "w", encoding="utf-8") as out_f:
        with open(input_file, "r", encoding="utf-8") as f:
            for idx, line in tqdm(enumerate(f, start=0), desc="Generating", total=total_lines):
                entry = json.loads(line)

                question, tests = _get_prompt_and_tests(entry, dataset)

                total_attempts += 1
                prompt = generate_prompt(question)

                last_code = ""
                passed = False

                for attempt in range(max_attempts):
                    gen = call_openrouter(prompt, api_key=api_key, model=api_model)
                    code = clean_generated_code(gen)
                    last_code = code

                    if _passes_tests(code, tests, dataset):
                        passed = True
                        entry["solutions"] = code
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        break

                if not passed:
                    failed_attempts += 1
                    entry["solutions"] = last_code
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    failure_ratio = failed_attempts / total_attempts if total_attempts > 0 else 0.0
    return {
        "dataset": dataset,
        "input_file": input_file,
        "output_file": output_file,
        "total": total_attempts,
        "failed": failed_attempts,
        "failure_ratio": failure_ratio,
        "max_attempts": max_attempts,
        "api_model": api_model,
    }




