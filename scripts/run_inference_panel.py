"""Launch the tkinter inference panel.

Usage:
  python scripts/run_inference_panel.py
  python scripts/run_inference_panel.py --checkpoint data/checkpoints/best.pt
  python scripts/run_inference_panel.py --checkpoint data/checkpoints/best.pt --device cuda
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui.inference_panel import main

if __name__ == "__main__":
    main()
