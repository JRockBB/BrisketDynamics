"""Pit Tesseract: lumped-parameter smoker ODE (Julia, ForwardDiff Jacobian VJP).

Wraps BrisketDynamics.jl's `pit_solve` / `pit_vjp`. Deliberately uses a
*different* differentiation strategy than the meat Tesseract: the 2-state
non-stiff ODE is differentiated with a chunked ForwardDiff dense Jacobian and
the VJP endpoint returns Jᵀv̄ — cheap and exact for a small system. The
composed pipeline is therefore a genuine continuous-adjoint + forward-mode
composition meeting only at the Tesseract VJP boundary.
"""

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    # Containerized: the api file sits at /tesseract/tesseract_api.py (no
    # grandparent). JULIA_PROJECT points at the baked-in package instead.
    REPO_ROOT = Path("/app/BrisketDynamics")

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

# In-container, PythonCall precompiles in a subprocess and cannot autodetect
# libpython; point it explicitly (mosaic topopt-jl pattern). No-op when the
# lib is not found (e.g. Windows dev host).
if not os.environ.get("JULIA_PYTHONCALL_LIB") and sys.platform == "linux":
    import glob as _glob
    import sysconfig as _sysconfig

    _libdir = _sysconfig.get_config_var("LIBDIR") or ""
    _libs = _glob.glob(os.path.join(_libdir, "libpython*.so.1.0")) or \
        _glob.glob(os.path.join(_libdir, "libpython*.so"))
    if _libs:
        os.environ["JULIA_PYTHONCALL_LIB"] = _libs[0]

import filelock

# All juliacall work runs on one dedicated thread, shared process-wide with the
# meat Tesseract (sys-level singleton) — jitted JAX callbacks arrive on XLA
# worker threads, and juliacall off its init thread deadlocks. See the meat
# tesseract_api.py docstring for the full story.
_EXEC_ATTR = "_brisket_julia_executor"
if getattr(sys, _EXEC_ATTR, None) is None:
    setattr(sys, _EXEC_ATTR, ThreadPoolExecutor(max_workers=1, thread_name_prefix="juliacall"))
_JL_THREAD: ThreadPoolExecutor = getattr(sys, _EXEC_ATTR)


# Direct mode for the containerized runtime — see meat tesseract_api.py.
_DIRECT = os.environ.get("BRISKET_JL_DIRECT", "0") == "1"


def _on_julia_thread(fn, *args):
    def _guarded():
        try:
            return fn(*args)
        except Exception as e:
            # Stringify HERE, on the Julia thread: a JuliaError crossing to
            # another thread deadlocks the moment traceback formatting calls
            # str() on it (juliacall off its init thread hangs). `from None`
            # keeps the original object out of the propagated chain.
            raise RuntimeError(f"Julia call failed: {type(e).__name__}: {e}") from None

    if _DIRECT:
        return _guarded()

    return _JL_THREAD.submit(_guarded).result()


def _init():
    global juliacall, jl
    import juliacall

    jl = juliacall.newmodule("PitTess")
    # In the container, JULIA_PROJECT points at the baked-in package copy;
    # locally it is unset and we activate the repo root.
    _project = os.environ.get("JULIA_PROJECT") or str(REPO_ROOT)
    jl.seval(f'import Pkg; Pkg.activate(raw"{_project}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jl.seval(
        """
        function _pit_apply(φ, u_knots, u_tknots, q_knots, q_tknots, tsave,
                            abstol, reltol, tpit0, t_amb, m0)
            cfg = PitConfig(Tpit0=tpit0, T_amb=t_amb, m0=m0)
            o = pit_solve(φ, u_knots, u_tknots, q_knots, q_tknots, tsave;
                          cfg=cfg, abstol=abstol, reltol=reltol)
            return o.Tpit, o.m_fuel, o.ignited
        end
        function _pit_vjp(φ, u_knots, u_tknots, q_knots, q_tknots, tsave,
                          abstol, reltol, tpit0, t_amb, m0, cT, cm)
            cfg = PitConfig(Tpit0=tpit0, T_amb=t_amb, m0=m0)
            g = pit_vjp(φ, u_knots, u_tknots, q_knots, q_tknots, tsave,
                        (Tpit=cT, m_fuel=cm); cfg=cfg, abstol=abstol, reltol=reltol)
            return g.dφ, g.du_knots, g.dqmeat_knots
        end
        """
    )


with filelock.FileLock(Path(tempfile.gettempdir()) / "brisket_julia_init.lock", timeout=600):
    _on_julia_thread(_init)


def _jvec(x) -> "juliacall.VectorValue":
    return juliacall.convert(jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64)))


class InputSchema(BaseModel):
    phi: Differentiable[Array[(7,), Float64]] = Field(
        description="Pit parameters [C_th J/K, UA W/K, r_max kg/s, dH_eff J/kg, "
                    "u_leak, k_spread 1/s, a0] (a0 = initial ignited bed fraction; "
                    "a0=1 disables ignition dynamics)"
    )
    u_knots: Differentiable[Array[(None,), Float64]] = Field(
        description="Damper schedule in [0,1] on the uniform grid `u_tknots`"
    )
    u_tknots: Array[(None,), Float64] = Field(description="Damper knot times [s]")
    qmeat_knots: Differentiable[Array[(None,), Float64]] = Field(
        description="Meat heat-load trajectory [W] on the uniform grid `q_tknots`"
    )
    q_tknots: Array[(None,), Float64] = Field(description="Heat-load knot times [s]")
    tsave: Array[(None,), Float64] = Field(description="Output save times [s]")
    abstol: float = Field(default=1e-8, description="ODE solver absolute tolerance")
    reltol: float = Field(default=1e-8, description="ODE solver relative tolerance")
    tpit0: float = Field(default=293.15, description="Initial pit temperature [K]")
    t_amb: float = Field(default=293.15, description="Ambient temperature [K]")
    m0: float = Field(default=7.0, description="Initial fuel load [kg]")


class OutputSchema(BaseModel):
    tpit: Differentiable[Array[(None,), Float64]] = Field(
        description="Pit gas temperature [K] at tsave"
    )
    m_fuel: Differentiable[Array[(None,), Float64]] = Field(
        description="Remaining fuel mass [kg] at tsave"
    )
    ignited: Array[(None,), Float64] = Field(
        description="Ignited bed fraction at tsave (diagnostic/state handoff)")


def _own(x) -> np.ndarray:
    """Eager owned copy made ON THE CALLBACK THREAD — XLA only guarantees
    callback input buffers during the invocation; see meat tesseract_api.py."""
    return np.array(x, dtype=np.float64, copy=True)


# Soft-fail mode (BRISKET_SOFT_FAIL=1): a failed solve returns sentinel
# outputs (absurd temperatures -> astronomically bad likelihood) instead of
# raising, so MCMC treats pathological parameter proposals as rejections
# rather than crashing the chain. NUTS warmup legitimately probes regions
# where the pit ODE is unsolvable (e.g. tiny C_th); everything outside
# sampling keeps hard failures.
SOFT_FAIL = os.environ.get("BRISKET_SOFT_FAIL", "0") == "1"
_SENTINEL_T = 1.0e4  # [K]


def apply(inputs: InputSchema) -> OutputSchema:
    phi, u, utk, q, qtk, tsave = (_own(inputs.phi), _own(inputs.u_knots),
                                  _own(inputs.u_tknots), _own(inputs.qmeat_knots),
                                  _own(inputs.q_tknots), _own(inputs.tsave))
    def _run():
        T, m, a = jl._pit_apply(
            _jvec(phi),
            _jvec(u),
            _jvec(utk),
            _jvec(q),
            _jvec(qtk),
            _jvec(tsave),
            float(inputs.abstol),
            float(inputs.reltol),
            float(inputs.tpit0),
            float(inputs.t_amb),
            float(inputs.m0),
        )
        # .copy() is load-bearing: np.asarray over a Julia vector is a
        # zero-copy VIEW of Julia memory; releasing it from another thread
        # can deadlock the whole process (see meat tesseract_api.py).
        return np.asarray(T).copy(), np.asarray(m).copy(), np.asarray(a).copy()

    try:
        T, m, a = _on_julia_thread(_run)
    except RuntimeError:
        if not SOFT_FAIL:
            raise
        nt = len(tsave)
        T, m, a = np.full(nt, _SENTINEL_T), np.zeros(nt), np.zeros(nt)
    return OutputSchema(tpit=T, m_fuel=m, ignited=a)


def abstract_eval(abstract_inputs):
    nt = abstract_inputs.tsave.shape[0]
    return {
        "tpit": ShapeDType(shape=(nt,), dtype="float64"),
        "m_fuel": ShapeDType(shape=(nt,), dtype="float64"),
        "ignited": ShapeDType(shape=(nt,), dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    phi, u, utk, q, qtk, tsave = (_own(inputs.phi), _own(inputs.u_knots),
                                  _own(inputs.u_tknots), _own(inputs.qmeat_knots),
                                  _own(inputs.q_tknots), _own(inputs.tsave))
    nt = len(tsave)
    cots = {name: (_own(cotangent_vector[name])
                   if name in cotangent_vector and name in vjp_outputs else None)
            for name in ("tpit", "m_fuel")}

    def _run():
        def cot(name):
            v = cots[name]
            return _jvec(v if v is not None else np.zeros(nt))

        dphi, du, dq = jl._pit_vjp(
            _jvec(phi),
            _jvec(u),
            _jvec(utk),
            _jvec(q),
            _jvec(qtk),
            _jvec(tsave),
            float(inputs.abstol),
            float(inputs.reltol),
            float(inputs.tpit0),
            float(inputs.t_amb),
            float(inputs.m0),
            cot("tpit"),
            cot("m_fuel"),
        )
        # .copy(): no zero-copy views of Julia memory may cross threads
        return (np.asarray(dphi).copy(), np.asarray(du).copy(),
                np.asarray(dq).copy())

    try:
        dphi, du, dq = _on_julia_thread(_run)
    except RuntimeError:
        if not SOFT_FAIL:
            raise
        # Zero gradients at an unsolvable point: the sampler is rejecting
        # this proposal anyway (sentinel likelihood); zeros keep HMC finite.
        dphi, du, dq = (np.zeros(len(phi)), np.zeros(len(u)), np.zeros(len(q)))
    out = {}
    if "phi" in vjp_inputs:
        out["phi"] = dphi
    if "u_knots" in vjp_inputs:
        out["u_knots"] = du
    if "qmeat_knots" in vjp_inputs:
        out["qmeat_knots"] = dq
    return out
