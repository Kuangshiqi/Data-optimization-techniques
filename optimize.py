from typing import Dict, Any, Callable
import argparse


def _normalize_dataset(dataset: str) -> str:
    d = (dataset or "").lower().strip()
    if d in {"mbpp", "apps", "codecontests"}:
        return d
    raise ValueError("dataset must be one of: mbpp / apps / codecontests")


def _normalize_tech(tech: str) -> str:
    t = (tech or "").strip()
    allowed = {"Aug", "Cle", "Ref", "Sel", "Syn"}
    if t not in allowed:
        raise ValueError("tech must be one of: Aug / Cle / Ref / Sel / Syn")
    return t


def run_tech(dataset: str, tech: str) -> Dict[str, Any]:
    dataset = _normalize_dataset(dataset)
    tech = _normalize_tech(tech)
    runners: Dict[str, Callable[[str], Dict[str, Any]]] = {
        "Aug": _run_aug,
        "Cle": _run_cle,
        "Ref": _run_ref,
        "Sel": _run_sel,
        "Syn": _run_syn,
    }

    return runners[tech](dataset)


def _run_aug(dataset: str) -> Dict[str, Any]:
    from .Techniques.Aug.run import augmentation
    return augmentation(dataset)


def _run_cle(dataset: str) -> Dict[str, Any]:
    from .Techniques.Cle.run import cleaning
    return cleaning(dataset)


def _run_ref(dataset: str) -> Dict[str, Any]:
    from .Techniques.Ref.run import refactoring
    return refactoring(dataset)


def _run_sel(dataset: str) -> Dict[str, Any]:
    from .Techniques.Sel.run import selection
    return selection(dataset)


def _run_syn(dataset: str) -> Dict[str, Any]:
    from .Techniques.Syn.run import synthesis
    return synthesis(dataset)


def main():
    parser = argparse.ArgumentParser(description="Run approach pipeline by dataset + tech")
    parser.add_argument("dataset", choices=["mbpp", "apps", "codecontests"])
    parser.add_argument("tech", choices=["Aug", "Cle", "Ref", "Sel", "Syn"])
    args = parser.parse_args()

    res = run_tech(dataset=args.dataset, tech=args.tech)
    print(res)


if __name__ == "__main__":
    main()