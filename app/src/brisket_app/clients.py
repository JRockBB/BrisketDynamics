"""Local (no-Docker) Tesseract clients for the two Julia components.

`Tesseract.from_tesseract_api` imports the API module in-process, so both
Tesseracts share one Julia runtime (separate juliacall modules over the same
BrisketDynamics package). This is the fast native dev loop; the containerized
path (`Tesseract.from_image`) is wired in for the packaged submission.
"""

import os
from pathlib import Path

# Must be set before jax / juliacall are imported anywhere in the process.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

from tesseract_core import Tesseract

REPO_ROOT = Path(__file__).resolve().parents[3]

_cache: dict[str, Tesseract] = {}


def meat_client() -> Tesseract:
    if "meat" not in _cache:
        _cache["meat"] = Tesseract.from_tesseract_api(
            REPO_ROOT / "tesseracts" / "meat" / "tesseract_api.py"
        )
    return _cache["meat"]


def pit_client() -> Tesseract:
    if "pit" not in _cache:
        _cache["pit"] = Tesseract.from_tesseract_api(
            REPO_ROOT / "tesseracts" / "pit" / "tesseract_api.py"
        )
    return _cache["pit"]
