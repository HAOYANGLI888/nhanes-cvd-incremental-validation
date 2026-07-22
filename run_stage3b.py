from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts" / "python"))

from stage3b_core import main


if __name__ == "__main__":
    raise SystemExit(main())
