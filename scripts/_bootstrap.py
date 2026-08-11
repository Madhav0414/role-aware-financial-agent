"""Make the scripts runnable from any working directory.

`python scripts/serve.py` puts scripts/ on sys.path, not the repository root, so
`import src` fails. Every data path in the project is written relative to the
root as well, so the working directory has to move there too -- otherwise the
imports resolve and the file reads do not, which is the more confusing failure.

Imported for its side effects, before any `src` import.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)
