from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from livermri_crossseq.cli.plan_anchor_pilot import main


if __name__ == "__main__":
    main()
