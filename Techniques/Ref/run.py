import json
from typing import Dict, Any

from tqdm import tqdm

from Autoflake import autoflake_clean
from Autopep8 import autopep8_clean
from Unify import unify_clean
from Docformatter import docformatter_clean


def refacoring(dataset: str) -> Dict[str, Any]:
    dataset = (dataset or "").lower().strip()
    if dataset not in {"mbpp", "apps", "codecontests"}:
        raise ValueError("dataset must be one of: mbpp / apps / codecontests (or cc)")

    input_file = f"Datasets/{dataset}/train_ori.jsonl"
    output_file = f"Datasets/{dataset}/train_ref.jsonl"
    start_index = 0

    n_total = 0
    n_written = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            n_total = len(lines)

            for idx in tqdm(
                range(start_index, n_total),
                desc="Refactoring code",
                total=n_total,
            ):
                entry = json.loads(lines[idx])
                code = entry.get("solutions", "")

                try:
                    code1 = autoflake_clean(code)
                except Exception:
                    code1 = code

                try:
                    code2 = autopep8_clean(code1)
                except Exception:
                    code2 = code1

                try:
                    code3 = docformatter_clean(code2)
                except Exception:
                    code3 = code2

                try:
                    code4 = unify_clean(code3)
                except Exception:
                    code4 = code3

                entry["code"] = code4
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n_written += 1

    return {
        "dataset": dataset,
        "input_file": input_file,
        "output_file": output_file,
        "start_index": start_index,
        "n_total": n_total,
        "n_written": n_written,
    }