from stage3a_core import build_parser, generate_denominator_tables, ensure_dirs, Stage3AContext, now
from pathlib import Path


def main():
    p = build_parser()
    args = p.parse_args()
    root = Path(args.project_root).resolve()
    ensure_dirs(root)
    ctx = Stage3AContext(root, root / "data/interim/stage2_harmonized_audit_dataset.csv", now(), root / "results/logs/stage3a_full_run.log")
    generate_denominator_tables(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
