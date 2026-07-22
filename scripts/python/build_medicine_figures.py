from __future__ import annotations

import shutil
from pathlib import Path

import build_manuscript_figures as figures


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "audit" / "medicine_figure_work"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    figures.OUT = OUT
    figures.SOURCE = OUT / "source_data"
    figures.SOURCE.mkdir(parents=True)

    figures.build_figure_2(embed_figure_text=False)
    figures.build_figure_3(embed_figure_text=False)
    figures.build_figure_4(embed_figure_text=False)
    figures.build_supplementary_figure_s1(embed_figure_text=False)
    print(f"medicine_figure_exports=PASS output={OUT}")


if __name__ == "__main__":
    main()
