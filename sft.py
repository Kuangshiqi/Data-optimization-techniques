import os
import argparse
from typing import Dict

from SFT.evaluation import run_infertest

PATHS: Dict[str, str] = {
    "dataset_root": "Dataset",

    "model_root": "Model",

    "results_root": "Results",
}


def dataset_to_train_file(dataset: str, tech: str) -> str:
    return os.path.join(PATHS["dataset_root"], dataset.upper(), tech, "train.jsonl")


def run_train(dataset: str, model: str, tech: str):
    import subprocess
    import sys

    train_data = dataset_to_train_file(dataset, tech)

    output_dir = os.path.join(PATHS["model_root"], dataset.lower(), model.lower(), tech.lower())
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "-m SFT.finetune"),
        "--model",
        model.lower(),
        "--data_path",
        train_data,
        "--output_dir",
        output_dir,
        "--num_train_epochs", "10",
        "--model_max_length", "1024",
        "--per_device_train_batch_size", "16",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "2e-5",
        "--logging_steps", "1",
        "--save_total_limit", "100",
        "--bf16", "False",
        "--lr_scheduler_type", "cosine",
        "--warmup_steps", "10",
        "--save_strategy", "epoch",
        "--report_to", "tensorboard",
    ]

    print("[run.py] train cmd:\n", " ".join(cmd))
    subprocess.check_call(cmd)


def run_eval(dataset: str, model: str, tech: str):
    res = run_infertest(dataset=dataset, model=model, tech=tech)
    print("[run.py] eval done:")
    print("  best_ckpt:", res["best_ckpt"])
    print("  val_pass1:", res["val_pass1"])
    print("  test_pass1:", res["test_pass1"])
    print("  avg_pass_ratio:", res["avg_pass_ratio"])
    print("  summary:", res["summary_path"])
    return res


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate with unified paths.")
    parser.add_argument("--dataset", required=True, choices=["apps", "codecontests", "mbpp"])
    parser.add_argument("--model", required=True, choices=["qwen1.5", "qwen3", "llama1", "llama3"])
    parser.add_argument("--tech", required=True, help="e.g. ori / aug / ...")
    parser.add_argument("--train", action="store_true", help="run training")
    parser.add_argument("--eval", dest="eval_", action="store_true", help="run evaluation")
    args = parser.parse_args()

    if not args.train and not args.eval_:
        args.train = True
        args.eval_ = True

    if args.train:
        run_train(args.dataset, args.model, args.tech)

    if args.eval_:
        run_eval(args.dataset, args.model, args.tech)


if __name__ == "__main__":
    main()