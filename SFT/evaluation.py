import os
import re
import gc
import json
import shutil
import torch
import tempfile
import subprocess
from typing import Dict, Tuple, Any, List, Callable

from tqdm import tqdm
from vllm import LLM, SamplingParams
from radon.metrics import mi_visit
from eval_code import eval_code

DEFAULTS = {
    # -------- 数据集--------
    "apps": {
        "valid_jsonl": "Dataset/APPS/valid.jsonl",
        "test_jsonl": "Dataset/APPS/test.jsonl",
    },
    "codecontests": {
        "valid_jsonl": "Dataset/CodeContests/valid.jsonl",
        "test_jsonl": "Dataset/CodeContests/test.jsonl",
    },
    "mbpp": {
        "valid_jsonl": "Dataset/MBPP/valid.jsonl",
        "test_jsonl": "Dataset/MBPP/test.jsonl",
    },

    # -------- checkpoint 根目录 --------
    # 约定：ckpt_dir = f"{saved_models_root}/{dataset}/{model}/{tech}"
    "saved_models_root": "Model/",

    # -------- 输出根目录 --------
    # 约定：out_dir = f"{results_root}/{dataset}/{model}/{tech}"
    "results_root": "Results/",

    # -------- vLLM 参数 --------
    "gpus": "0",
    "tensor": 1,
    "memory": 0.9,
    "temperature": 0.1,
    "top_p": 0.95,
    "batch_size": 64,
    "max_tokens": 1024,
    "num_return_sequences": 1,

    # -------- 质量评估参数 --------
    "metrics_k": 1,
    "pylint_timeout": 30,
}



def get_eval_fns_for_dataset(dataset: str) -> Tuple[Callable[..., float], Callable[..., float]]:
    """
    返回 (pass1_fn, apr_fn)
    - pass1_fn(code_file, data_file, k) -> float
    - apr_fn(code_file, data_file, k) -> float
    """
    dataset = (dataset or "").lower()

    if dataset == "mbpp":
        from test_mbpp import eval_code as mbpp_eval_code

        def pass1(code_file: str, data_file: str, k: int = 1) -> float:
            all_code: Dict[str, str] = {}
            with open(code_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    all_code[entry["id"]] = entry["code"]

            with open(data_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            total_cases = len(lines)
            total_passed = 0

            for idx, line in tqdm(
                enumerate(lines), total=total_cases, desc=f"[MBPP] pass@1 {os.path.basename(data_file)} k={k}"
            ):
                item = json.loads(line)
                tests = item.get("test_list")

                ok = False
                for j in range(k):
                    eid = f"{idx}_{j}"
                    code = all_code.get(eid)
                    if code and float(mbpp_eval_code(code, tests)) == 1.0:
                        ok = True
                        break
                if ok:
                    total_passed += 1

            return total_passed / total_cases if total_cases > 0 else 0.0

        def apr(code_file: str, data_file: str, k: int = 1) -> float:
            all_code: Dict[str, str] = {}
            with open(code_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    all_code[entry["id"]] = entry["code"]

            with open(data_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            total_cases = len(lines)
            ratios: List[float] = []

            for idx, line in tqdm(
                enumerate(lines), total=total_cases, desc=f"[MBPP] APR {os.path.basename(data_file)} k={k}"
            ):
                item = json.loads(line)
                tests = item.get("test_list")

                best_ratio = 0.0
                for j in range(k):
                    eid = f"{idx}_{j}"
                    code = all_code.get(eid)
                    if not code:
                        continue
                    try:
                        ratio = float(mbpp_eval_code(code, tests))
                    except Exception:
                        ratio = 0.0
                    best_ratio = max(best_ratio, ratio)
                ratios.append(best_ratio)

            return (sum(ratios) / len(ratios)) if ratios else 0.0

        return pass1, apr
    else:
        def pass1(code_file: str, data_file: str, k: int = 1) -> float:
            all_code: Dict[str, str] = {}
            with open(code_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    all_code[entry["id"]] = entry["code"]

            with open(data_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            total_cases = len(lines)
            total_passed = 0

            for idx, line in tqdm(
                enumerate(lines), total=total_cases, desc=f"[{dataset.upper() or 'GEN'}] pass@1 {os.path.basename(data_file)} k={k}"
            ):
                item = json.loads(line)
                in_outs = item.get("input_output")
                if isinstance(in_outs, str):
                    in_outs = json.loads(in_outs)

                ok = False
                for j in range(k):
                    eid = f"{idx}_{j}"
                    code = all_code.get(eid)
                    if code and float(eval_code(in_outs, code)) == 1.0:
                        ok = True
                        break
                if ok:
                    total_passed += 1

            return total_passed / total_cases if total_cases > 0 else 0.0

        def apr(code_file: str, data_file: str, k: int = 1) -> float:
            all_code: Dict[str, str] = {}
            with open(code_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    all_code[entry["id"]] = entry["code"]

            with open(data_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            total_cases = len(lines)
            ratios: List[float] = []

            for idx, line in tqdm(
                enumerate(lines), total=total_cases, desc=f"[{dataset.upper() or 'GEN'}] APR {os.path.basename(data_file)} k={k}"
            ):
                item = json.loads(line)
                in_outs = item.get("input_output")
                if isinstance(in_outs, str):
                    in_outs = json.loads(in_outs)

                best_ratio = 0.0
                for j in range(k):
                    eid = f"{idx}_{j}"
                    code = all_code.get(eid)
                    if not code:
                        continue
                    try:
                        ratio = float(eval_code(in_outs, code))
                    except Exception:
                        ratio = 0.0
                    best_ratio = max(best_ratio, ratio)
                ratios.append(best_ratio)

            return (sum(ratios) / len(ratios)) if ratios else 0.0

        return pass1, apr


# -----------------------------
# vLLM 生成相关
# -----------------------------
def build_instruction_prompt(instruction: str) -> str:
    return f"""  
You are an AI programming assistant, please generate Python code according to the following instruction.
### Instruction:
{(instruction or '').strip()}
### Response:
""".lstrip()

def extract_code_from_response(text: str) -> str:
    if not text:
        return ""
    if "```python" in text:
        text = text.split("```python", 1)[1].rsplit("```", 1)[0]
    elif "```" in text:
        # 兜底：任何 fence
        text = text.split("```", 1)[0]
    if "</s>" in text:
        text = text.split("</s>", 1)[0]
    return text.strip()


def setup_runtime_env(gpus: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.empty_cache()


def list_checkpoints(base_ckpt_dir: str) -> List[str]:
    return sorted(
        [d for d in os.listdir(base_ckpt_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )


def generate_for_ckpt(
    *,
    ckpt_path: str,
    data_file: str,
    out_file: str,
    temperature: float,
    memory: float,
    num_return_sequences: int,
    tensor: int,
    batch_size: int,
    top_p: float,
    max_tokens: int,
):
    llm = LLM(
        model=ckpt_path,
        tensor_parallel_size=tensor,
        trust_remote_code=True,
        gpu_memory_utilization=memory,
    )
    try:
        data = [json.loads(l) for l in open(data_file, "r", encoding="utf-8")]
        with open(out_file, "w", encoding="utf-8") as outf:
            for start in tqdm(
                range(0, len(data), batch_size),
                desc=f"Gen {os.path.basename(ckpt_path)}",
                unit="batch",
            ):
                batch = data[start : start + batch_size]

                prompts: List[str] = []
                for item in batch:
                    prompt_val = item["question"]
                    prompts.append(build_instruction_prompt(str(prompt_val)))

                sampling = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=num_return_sequences,
                )
                outputs = llm.generate(prompts, sampling_params=sampling)

                for i, out in enumerate(outputs):
                    idx_global = start + i
                    codes = [extract_code_from_response(o.text) for o in out.outputs]
                    for j, code in enumerate(codes):
                        outf.write(
                            json.dumps({"id": f"{idx_global}_{j}", "code": code}, ensure_ascii=False) + "\n"
                        )
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def select_best_checkpoint(
    *,
    dataset: str,
    base_ckpt_dir: str,
    valid_jsonl: str,
    temperature: float,
    memory: float,
    tensor: int,
    num_return_sequences: int,
    batch_size: int,
    top_p: float,
    max_tokens: int,
) -> Tuple[str, float]:
    ckpts = list_checkpoints(base_ckpt_dir)
    tmp_dir = tempfile.mkdtemp(prefix="val_gen_")
    try:
        pass1_fn, _apr_fn = get_eval_fns_for_dataset(dataset)

        val_scores: Dict[str, float] = {}
        for ck in ckpts:
            ckpt_path = os.path.join(base_ckpt_dir, ck)
            tmp_file = os.path.join(tmp_dir, f"all_code_{ck}.jsonl")

            generate_for_ckpt(
                ckpt_path=ckpt_path,
                data_file=valid_jsonl,
                out_file=tmp_file,
                temperature=temperature,
                memory=memory,
                num_return_sequences=num_return_sequences,
                tensor=tensor,
                batch_size=batch_size,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            p1 = pass1_fn(tmp_file, valid_jsonl, k=1)
            val_scores[ck] = p1

        best_ckpt = max(val_scores, key=val_scores.get)
        return best_ckpt, val_scores[best_ckpt]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# -----------------------------
# 代码质量指标（pylint / MI）
# -----------------------------
def run_pylint_score(code_str: str, timeout: int = 30) -> float:
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
        tmp_file.write(code_str or "")
        tmp_file.flush()
        tmp_file.close()

        result = subprocess.run(
            ["pylint", tmp_file.name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        m = re.search(r"rated at\s*([0-9]+\.?[0-9]*)/10", output)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0
    finally:
        if tmp_file:
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


def compute_mi_score(code_str: str) -> float:
    try:
        return float(mi_visit(code_str or "", True))
    except Exception:
        return 0.0


def load_jsonl(path: str) -> List[dict]:
    data: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def run_infertest(
    *,
    dataset: str,
    model: str,
    tech: str,
) -> Dict[str, Any]:
    dataset = dataset.lower().strip()
    model = model.lower().strip()
    tech = tech.lower().strip()

    if dataset not in DEFAULTS:
        raise ValueError(f"Unknown dataset: {dataset}. expected one of: apps/codecontests/mbpp")

    gpus = DEFAULTS["gpus"]
    tensor = DEFAULTS["tensor"]
    memory = DEFAULTS["memory"]
    temperature = DEFAULTS["temperature"]
    top_p = DEFAULTS["top_p"]
    batch_size = DEFAULTS["batch_size"]
    max_tokens = DEFAULTS["max_tokens"]
    num_return_sequences = DEFAULTS["num_return_sequences"]
    metrics_k = DEFAULTS["metrics_k"]
    pylint_timeout = DEFAULTS["pylint_timeout"]

    valid_jsonl = DEFAULTS[dataset]["valid_jsonl"]
    test_jsonl = DEFAULTS[dataset]["test_jsonl"]

    base_ckpt_dir = os.path.join(DEFAULTS["saved_models_root"], dataset, model, tech)
    out_dir = os.path.join(DEFAULTS["results_root"], dataset, model, tech)
    os.makedirs(out_dir, exist_ok=True)

    setup_runtime_env(gpus)

    best_ckpt, best_val = select_best_checkpoint(
        dataset=dataset,
        base_ckpt_dir=base_ckpt_dir,
        valid_jsonl=valid_jsonl,
        temperature=temperature,
        memory=memory,
        tensor=tensor,
        num_return_sequences=num_return_sequences,
        batch_size=batch_size,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    best_ckpt_path = os.path.join(base_ckpt_dir, best_ckpt)
    test_code_file = os.path.join(out_dir, "all_code.jsonl")

    generate_for_ckpt(
        ckpt_path=best_ckpt_path,
        data_file=test_jsonl,
        out_file=test_code_file,
        temperature=temperature,
        memory=memory,
        num_return_sequences=num_return_sequences,
        tensor=tensor,
        batch_size=batch_size,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    pass1_fn, apr_fn = get_eval_fns_for_dataset(dataset)
    test_pass1 = pass1_fn(test_code_file, test_jsonl, k=1)
    avg_pass_ratio = apr_fn(test_code_file, test_jsonl, k=metrics_k)

    entries = load_jsonl(test_code_file)
    codes = [rec.get("code", "") or "" for rec in entries]

    pylint_scores = [
        run_pylint_score(code, timeout=pylint_timeout)
        for code in tqdm(codes, desc=f"[{dataset}/{model}/{tech}] pylint")
    ]
    avg_pylint = sum(pylint_scores) / len(pylint_scores) if pylint_scores else 0.0

    mi_scores = [compute_mi_score(code) for code in tqdm(codes, desc=f"[{dataset}/{model}/{tech}] MI")]
    avg_mi = sum(mi_scores) / len(mi_scores) if mi_scores else 0.0

    summary_path = os.path.join(out_dir, "test.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset}\n")
        f.write(f"model={model}\n")
        f.write(f"tech={tech}\n")
        f.write(f"best_ckpt={best_ckpt}\n")
        f.write(f"val_pass1={best_val:.4f}\n")
        f.write(f"test_pass1={test_pass1:.4f}\n")
        f.write(f"avg_pass_ratio={avg_pass_ratio}\n")
        f.write(f"avg_pylint_score={avg_pylint}\n")
        f.write(f"avg_mi_score={avg_mi}\n")

    return {
        "dataset": dataset,
        "model": model,
        "tech": tech,
        "base_ckpt_dir": base_ckpt_dir,
        "out_dir": out_dir,
        "best_ckpt": best_ckpt,
        "best_ckpt_path": best_ckpt_path,
        "val_pass1": best_val,
        "test_pass1": test_pass1,
        "avg_pass_ratio": avg_pass_ratio,
        "avg_pylint_score": avg_pylint,
        "avg_mi_score": avg_mi,
        "test_code_file": test_code_file,
        "summary_path": summary_path,
    }