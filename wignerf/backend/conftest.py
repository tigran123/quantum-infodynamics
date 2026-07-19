import sys
from pathlib import Path

# Make `core` importable regardless of pytest's rootdir/import mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
