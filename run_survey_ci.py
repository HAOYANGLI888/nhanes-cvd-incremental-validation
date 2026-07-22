#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked NHANES survey-aware confidence intervals.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    scripts = project_root / "scripts" / "python"
    sys.path.insert(0, str(scripts))
    from survey_ci_core import run_survey_ci

    return run_survey_ci(project_root, resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
