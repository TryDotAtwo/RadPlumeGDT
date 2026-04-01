from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rad_plume.download_medium_range_box import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ECMWF medium-range data for the project domain.")
    parser.add_argument("--demo", action="store_true", help="Short download demo instead of the full horizon.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(demo=args.demo)
