import json
import random
from typing import Dict, Callable, Any, List, Optional

from tqdm import tqdm

from ConditionalExpToSingleIf import conditional_to_single_if
from ForToWhile import for_to_while
from IfDividing import if_dividing
from InfixExpressionDividing import divide_infix_expressions
from ReverseIfElse import reverse_if_else
from SingleIfToConditionalExp import single_if_to_conditional
from StatementsOrderRearrangement import rearrange_code
from WhileToFor import while_to_for
from LocalVarRename import rename_local_variables
from AddAssignmentToEqualAssignment import add_assign_to_equal_assign
from VarDeclarationMerging import var_declaration_merging
from VarDeclarationDividing import var_declaration_dividing
from origintrans import origin_trans

from test_mbpp import eval_code as eval_code_mbpp
from eval_code import eval_code as eval_code_apps_cc


TRANSFORMS: Dict[str, Callable[[str], str]] = {
    "conditional_to_single_if": conditional_to_single_if,
    "single_if_to_conditional": single_if_to_conditional,
    "while_to_for": while_to_for,
    "for_to_while": for_to_while,
    "if_dividing": if_dividing,
    "divide_infix_expressions": divide_infix_expressions,
    "reverse_if_else": reverse_if_else,
    "rearrange_code": rearrange_code,
    "rename_local_variables": rename_local_variables,
    "add_assign_to_equal_assign": add_assign_to_equal_assign,
    "var_declaration_merging": var_declaration_merging,
    "var_declaration_dividing": var_declaration_dividing,
}


def _detect_dataset(dataset: str) -> str:
    d = (dataset or "").lower().strip()
    if d in {"mbpp", "apps", "codecontests"}:
        return d
    raise ValueError("dataset must be one of: mbpp / apps / codecontests (or cc)")


def _get_tests_for_entry(entry: Dict[str, Any], dataset: str):
    if dataset == "mbpp":
        return entry.get("test_list")
    in_outs = entry.get("input_output")
    if isinstance(in_outs, str):
        try:
            in_outs = json.loads(in_outs)
        except Exception:
            pass
    return in_outs


def _passes_tests(modified_code: str, tests: Any, dataset: str) -> bool:
    # 要求：mbpp 与 apps/cc 分开测试流程
    if dataset == "mbpp":
        try:
            return float(eval_code_mbpp(modified_code, tests)) == 1.0
        except Exception:
            return False

    # APPS/CodeContests
    try:
        return float(eval_code_apps_cc(tests, modified_code)) == 1.0
    except Exception:
        return False


def augmentation(dataset: str) -> Dict[str, Any]:
    dataset = _detect_dataset(dataset)

    input_file = f"Datasets/{dataset}/train_ori.jsonl"
    output_file = f"Datasets/{dataset}/train_aug.jsonl"

    seed = 42
    keep_original = True
    verbose = True
    transforms: Optional[Dict[str, Callable[[str], str]]] = None

    transforms = transforms or TRANSFORMS
    random.seed(seed)

    all_entries: List[str] = []
    n_input = 0
    n_added = 0

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        n_input = len(lines)

        if keep_original:
            for line in tqdm(lines, desc="Collecting Original", total=len(lines)):
                all_entries.append(line.strip())

        for idx in tqdm(range(len(lines)), desc="Applying Transforms", total=len(lines)):
            entry = json.loads(lines[idx])
            code = entry["solutions"]
            tests = _get_tests_for_entry(entry, dataset)

            try:
                origin = origin_trans(code)
            except Exception:
                origin = code

            added_count = 0

            for _name, transform_fn in transforms.items():
                try:
                    modified_code = transform_fn(code)
                except Exception:
                    modified_code = origin

                if modified_code != origin and _passes_tests(modified_code, tests, dataset):
                    new_entry = entry.copy()
                    new_entry["solutions"] = modified_code
                    all_entries.append(json.dumps(new_entry, ensure_ascii=False))
                    added_count += 1
                    n_added += 1

            if verbose:
                print(f"[Entry {idx}] 增加了 {added_count} 条新数据")

    random.shuffle(all_entries)
    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in all_entries:
            out_f.write(line + "\n")

    if verbose:
        print("Processing completed with shuffled output!")

    return {
        "dataset": dataset,
        "input_file": input_file,
        "output_file": output_file,
        "n_input_lines": n_input,
        "n_output_lines": len(all_entries),
        "n_added": n_added,
        "seed": seed,
        "keep_original": keep_original,
        "transforms": list((transforms or {}).keys()),
    }