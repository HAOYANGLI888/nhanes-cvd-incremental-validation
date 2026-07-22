from stage3a_core import build_parser, read_frozen_features
from pathlib import Path


def main():
    args = build_parser().parse_args()
    root = Path(args.project_root).resolve()
    read_frozen_features(root)
    print("Preprocessing specifications loaded from frozen variables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
