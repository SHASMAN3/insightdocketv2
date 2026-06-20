"""
Root conftest.py — ensures the project root is on sys.path so that
`import app` works from any test file without installing the package.
"""

import sys
from pathlib import Path

# Add the project root (parent of this file) to sys.path
sys.path.insert(0, str(Path(__file__).parent))
