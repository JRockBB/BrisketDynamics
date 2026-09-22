"""One message when Julia cannot start, instead of one setup error per Julia-backed test.

Seven test modules drive the monolith or the slab2d Tesseract in-process through juliacall. When that runtime
cannot initialize (on Windows the juliacall project's own dependencies have been found uninstantiated: PythonCall
fails to precompile and the Tesseract API cannot load), every one of their tests errored at setup with the same
stack. Here the first Julia-backed test to run probes the runtime once; if the probe fails, that test FAILS with
the cause and the repair, and the rest of the Julia-backed tests are skipped with a one-line reason. The suite
stays red (nothing is hidden), and the message is read once.
"""

from __future__ import annotations

import pytest

JULIA_MODULES = {
    "test_composed_grad", "test_duration_grad", "test_monolith_bridge", "test_monolith_tesseract",
    "test_mpc", "test_planner_decisions", "test_slab2d_tesseract",
}
REPAIR = ("the Julia runtime behind the app could not start; on Windows the usual cause is juliacall's own project "
          "left uninstantiated: run  uv run python -c \"import juliapkg; print(juliapkg.project())\"  and then "
          "julia --project=<that path> -e 'using Pkg; Pkg.instantiate()'  (docs/install.md, Windows)")
_probe: dict[str, Exception | None] = {}


def _julia_probe() -> Exception | None:
    if "result" not in _probe:
        try:
            from brisket_app.clients import monolith_client

            monolith_client()
            _probe["result"] = None
        except Exception as e:  # noqa: BLE001 - the cause is reported, whatever it is
            _probe["result"] = e
            _probe["reported"] = False
    return _probe["result"]


def pytest_runtest_setup(item):
    if item.module.__name__.rsplit(".", 1)[-1] not in JULIA_MODULES:
        return
    err = _julia_probe()
    if err is None:
        return
    if not _probe.get("reported"):
        _probe["reported"] = True
        pytest.fail(f"{REPAIR}\n\ncause: {type(err).__name__}: {err}", pytrace=False)
    pytest.skip(f"Julia runtime unavailable ({type(err).__name__}); see the first failure")
