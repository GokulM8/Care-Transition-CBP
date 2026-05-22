from pathlib import Path
import sys


dashboard_dir = Path(__file__).resolve().parent / "dashboard"
if str(dashboard_dir) not in sys.path:
    sys.path.insert(0, str(dashboard_dir))


import app as _dashboard_app  # noqa: F401