import os
import sys

# Ensure the repository root (where the modules under test live) is importable
# regardless of pytest's rootdir / invocation directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
