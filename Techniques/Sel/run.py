import json
import os
from typing import Dict, Any, List, Optional

import torch
from tqdm import tqdm
from style_ranker.rank import rank_and_filter


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def selection(dataset: str) -> Dict[str, Any]:
    dataset = (dataset or "").lower().strip()
    if dataset not in {"mbpp", "apps", "codecontests"}:
        raise ValueError("dataset must be one of: mbpp / apps / codecontests (or cc)")
    if dataset == "cc":
        dataset = "codecontests"

    input_file = f"Datasets/{dataset}/train_ori.jsonl"
    output_file = f"Datasets/{dataset}/train_sel.jsonl"
    model_path = "lizhuang144/scar-gte-base"
    ratio: Optional[float] = 0.2
    topk: Optional[int] = None
    threshold: Optional[float] = None
    max_length = 512
    cuda_visible_devices = "0"


    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} samples from {input_file}")

    instructions = [item["question"] for item in data]
    answers = [item["solutions"] for item in data]

    print("Scoring and filtering...")
    filtered = rank_and_filter(
        model_path,
        instructions,
        answers,
        max_length=max_length,
        device=device,
        topk=topk,
        threshold=threshold,
        ratio=ratio,
    )


    print(f"Saving {len(filtered)} filtered samples to {output_file}")
    n_written = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for instr, ans, score in tqdm(filtered, desc="Writing"):
            json.dump({"question": instr, "solutions": ans, "score": score}, f, ensure_ascii=False)
            f.write("\n")
            n_written += 1

    return {
        "dataset": dataset,
        "input_path": input_file,
        "output_path": output_file,
        "model_path": model_path,
        "ratio": ratio,
        "topk": topk,
        "threshold": threshold,
        "max_length": max_length,
        "cuda_visible_devices": cuda_visible_devices,
        "n_in": len(data),
        "n_out": n_written,
    }
