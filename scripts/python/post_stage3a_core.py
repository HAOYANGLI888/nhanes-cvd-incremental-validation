from __future__ import annotations

from pathlib import Path

from class_imbalance_sensitivity import run_class_imbalance
from dca_module import run_dca
from final_freeze_module import run_final_freeze
from meta_analysis_module import run_meta_analysis
from post_stage3a_common import log
from uacr_sensitivity import run_uacr


ALLOWED_MODULES = ["uacr", "class_imbalance", "meta_analysis", "dca", "final_freeze"]


def run_modules(project_root: Path, modules: list[str], resume: bool, workers: int) -> int:
    unknown = [module for module in modules if module not in ALLOWED_MODULES]
    if unknown:
        raise ValueError(f"Unknown modules: {unknown}")
    project_root = Path(project_root).resolve()
    log_path = project_root / "results" / "logs" / "post_stage3a_full_run.log"
    log(log_path, f"Post-Stage-3A run starting; modules={modules}; resume={resume}; workers={workers}")
    for module in modules:
        log(log_path, f"Module starting: {module}")
        if module == "uacr":
            status = run_uacr(project_root, resume, workers, log_path)
        elif module == "class_imbalance":
            status = run_class_imbalance(project_root, resume, workers, log_path)
        elif module == "meta_analysis":
            status = run_meta_analysis(project_root)
        elif module == "dca":
            status = run_dca(project_root)
        else:
            status = run_final_freeze(project_root)
        log(log_path, f"Module finished: {module}; gate={status}")
        if status != "PASS":
            return 2
    log(log_path, "Post-Stage-3A run finished; all requested module gates=PASS")
    return 0
