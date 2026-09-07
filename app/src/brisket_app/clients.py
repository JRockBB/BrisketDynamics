"""Local (no-Docker) client for the monolith Tesseract.

`Tesseract.from_tesseract_api` imports the API module in-process (one Julia
runtime, one juliacall module over the BrisketDynamics package). This is the fast native dev loop; the containerized
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
_WRAPPED_METHODS = (
    "apply",
    "vector_jacobian_product",
    "jacobian_vector_product",
    "jacobian",
    "abstract_eval",
)


def _serialize_calls(t: Tesseract) -> Tesseract:
    """Wrap every endpoint method of `t` with the process-wide call lock.

    Parameters
    ----------
    t : Tesseract
        Client whose endpoint methods get serialized in place.

    Returns
    -------
    Tesseract
        The same client, methods wrapped (see `_CALL_LOCK` comment).
    """
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


def monolith_client() -> Tesseract:
    """Return the process-wide monolith Tesseract client (created on first call).

    Returns
    -------
    Tesseract
        In-process client for the whole coupled cook (pit + meat + humidity in
        one solve; forward-mode jvp / vjp / jacobian endpoints), call-serialized.
    """
    if "monolith" not in _cache:
        _cache["monolith"] = _serialize_calls(
            Tesseract.from_tesseract_api(
                REPO_ROOT / "tesseracts" / "monolith" / "tesseract_api.py"
            )
        )
    return _cache["monolith"]


def slab2d_client() -> Tesseract:
    """Return the process-wide 2-D (r-z) Tesseract client (created on first call).

    Returns
    -------
    Tesseract
        In-process client for the 2-D coupled cook — the fidelity rung that
        shadow-scores 1-D plans (true cold spot, bark distribution) —
        call-serialized like the monolith.
    """
    if "slab2d" not in _cache:
        _cache["slab2d"] = _serialize_calls(
            Tesseract.from_tesseract_api(
                REPO_ROOT / "tesseracts" / "slab2d" / "tesseract_api.py"
            )
        )
    return _cache["slab2d"]
