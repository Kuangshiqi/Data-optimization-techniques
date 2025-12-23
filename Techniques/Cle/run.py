import os
import sys
import subprocess
from typing import Dict, Any


def cleaning(dataset: str) -> Dict[str, Any]:
    dataset = (dataset or "").lower().strip()
    if dataset not in {"mbpp", "apps", "codecontests"}:
        raise ValueError("dataset must be one of: mbpp / apps / codecontests")

    cle_dir = os.path.dirname(__file__)

    training_set_json = "Approach/Cle/training_set.json"
    predictions_json = f"Datasets/{dataset}/train_ori.jsonl"

    number_of_batches = "1"
    semgrep_result_prefix = predictions_json[:-5] + "_semgrep_result_batch" if predictions_json.endswith(".json") else "xxx_semgrep_result_batch"

    install_semgrep = True

    # 1) pip install semgrep
    if install_semgrep:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "semgrep"])

    # 2) analyze_code.py <predictions>.json
    subprocess.check_call([sys.executable, os.path.join(cle_dir, "analyze_code.py"), predictions_json])

    # 3) process_results.py <prefix> <number_of_batches>
    subprocess.check_call(
        [sys.executable, os.path.join(cle_dir, "process_results.py"), semgrep_result_prefix, str(number_of_batches)]
    )
    # 4) clean_dataset.py <training_set>.json
    subprocess.check_call([sys.executable, os.path.join(cle_dir, "clean_dataset.py"), training_set_json])

    return {
        "dataset": dataset,
        "training_set_json": training_set_json,
        "predictions_json": predictions_json,
        "semgrep_result_prefix": semgrep_result_prefix,
        "number_of_batches": number_of_batches,
        "install_semgrep": install_semgrep,
    }