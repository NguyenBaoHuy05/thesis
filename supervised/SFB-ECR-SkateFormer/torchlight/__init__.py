import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent / "torchlight"
if _PKG_DIR.is_dir() and str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

try:
    from torchlight import DictAction, IO, import_class, str2bool
except ImportError:
    pass
