from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rad_plume.config import SETTINGS
from rad_plume.main import create_run_output_dir, run_summary_map


if __name__ == "__main__":
    run_summary_map(SETTINGS, None, create_run_output_dir())
