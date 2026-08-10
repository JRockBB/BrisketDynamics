"""Local (no-Docker) Tesseract clients for the two Julia components.

`Tesseract.from_tesseract_api` imports the API module in-process, so both
Tesseracts share one Julia runtime (separate juliacall modules over the same
BrisketDynamics package). This is the fast native dev loop; the containerized
path (`Tesseract.from_image`) is wired in for the packaged submission.
"""

import functools
import os
import threading
from pathlib import Path

# Must be set before jax / juliacall are imported anywhere in the process.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

from tesseract_core import Tesseract

REPO_ROOT = Path(__file__).resolve().parents[3]

_cache: dict[str, Tesseract] = {}

# One process-wide lock serializing EVERY Tesseract endpoint call, input
# packing included. JAX can execute the primal (apply) and tangent (jvp)
# dispatches of one differentiated call concurrently on XLA worker threads,
# and the LocalClient's packing path is not thread-safe — observed as a
# flaky BoundsError where one call's scalar landed in another call's
# trajectory slot. The Julia side is already serialized by the tesseract-api
# executor; this extends the serialization to the Python packing layer.
# Cost: none in practice (there is only one Julia thread anyway).
_CALL_LOCK = threading.Lock()
_WRAPPED_METHODS = ("apply", "vector_jacobian_product", "jacobian_vector_product",
                    "jacobian", "abstract_eval")


def _serialize_calls(t: Tesseract) -> Tesseract:
    for name in _WRAPPED_METHODS:
        fn = getattr(t, name, None)
        if fn is None:
            continue

        @functools.wraps(fn)
        def locked(*args, _fn=fn, **kwargs):
            with _CALL_LOCK:
                return _fn(*args, **kwargs)

        setattr(t, name, locked)
    return t


def meat_client() -> Tesseract:
    if "meat" not in _cache:
        _cache["meat"] = _serialize_calls(Tesseract.from_tesseract_api(
            REPO_ROOT / "tesseracts" / "meat" / "tesseract_api.py"
        ))
    return _cache["meat"]


def pit_client() -> Tesseract:
    if "pit" not in _cache:
        _cache["pit"] = _serialize_calls(Tesseract.from_tesseract_api(
            REPO_ROOT / "tesseracts" / "pit" / "tesseract_api.py"
        ))
    return _cache["pit"]
