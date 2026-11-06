"""HinglishSense backend package.

Makes the shared research package (python/src/hinglish) importable so the API
uses the exact same config loader, label map, normalization, dataset-style
transform and calibration code as training.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(os.environ.get("HINGLISH_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
os.environ.setdefault("HINGLISH_PROJECT_ROOT", str(_ROOT))
_SRC = str(_ROOT / "python" / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
