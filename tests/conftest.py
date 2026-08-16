"""
Put the repository root on sys.path.

The project modules live at the top level rather than in an installed package,
so without this the suite only collects when pytest happens to be invoked as
``python -m pytest`` from the repository root.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent.absolute()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
