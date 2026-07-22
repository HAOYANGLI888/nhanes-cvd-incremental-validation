#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run locked post-Stage-3A sensitivity modules.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--modules", nargs="+", default=["uacr", "class_imbalance", "meta_analysis", "dca", "final_freeze"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "scripts" / "python"))
    from post_stage3a_core import run_modules

    return run_modules(root, args.modules, resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
