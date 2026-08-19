"""Meat Tesseract: 1D heat-moisture brisket PDE (Julia, SciMLSensitivity adjoint).

Wraps BrisketDynamics.jl's `meat_solve` / `meat_vjp`. The VJP endpoint is backed
by a continuous adjoint (InterpolatingAdjoint + re-taped ReverseDiff VJP) of
the stiff method-of-lines solve — to our knowledge the first public
SciMLSensitivity adjoint exposed through the Tesseract VJP interface.

Julia bridge pattern follows pasteurlabs/mosaic `topopt-jl`: synchronous
module-level init guarded by a cross-process file lock (uvicorn worker
healthchecks SIGKILL slow async inits).

Threading: ALL juliacall work (init, conversions, solves) runs on one dedicated
thread, shared process-wide by both Tesseracts via a sys-level singleton.
Callers from any thread block on a future. This matters because JAX executes
jitted callbacks on XLA worker threads, and juliacall invoked off the Julia
init thread deadlocks (observed: numpyro SVI hung with 0% CPU; eager-mode
calls from the main thread never hit it).
"""

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType

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

_EXEC_ATTR = "_brisket_julia_executor"
if getattr(sys, _EXEC_ATTR, None) is None:
    setattr(sys, _EXEC_ATTR, ThreadPoolExecutor(max_workers=1, thread_name_prefix="juliacall"))
_JL_THREAD: ThreadPoolExecutor = getattr(sys, _EXEC_ATTR)


# Direct mode (BRISKET_JL_DIRECT=1, set in the container env): run Julia on
# the CALLING thread with no executor. The executor exists to serialize
# XLA-worker-thread callbacks under tesseract-jax's LocalClient; inside the
# containerized runtime there is no JAX, and owning Julia on a non-main
# thread hangs juliacall's atexit (jl_atexit_hook from the main thread).
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

    jl = juliacall.newmodule("MeatTess")
    # In the container, JULIA_PROJECT points at the baked-in package copy;
    # locally it is unset and we activate the repo root.
    _project = os.environ.get("JULIA_PROJECT") or str(REPO_ROOT)
    jl.seval(f'import Pkg; Pkg.activate(raw"{_project}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jl.seval(
        """
        function _meat_apply(θ, knots, tknots, tsave, abstol, reltol, t0,
                             T0f, W0f, C0f, t_wrap)
            cfg = MeatConfig(T0=t0)
            o = meat_solve(θ, knots, tknots, tsave; cfg=cfg, abstol=abstol,
                           reltol=reltol, T0f=T0f, W0f=W0f, C0f=C0f,
                           t_wrap=t_wrap)
            return (o.Tsurf, o.Tcore, o.j_evap, o.q_meat, o.C_field_final,
                    o.w_retained, o.T_field[:, end], o.W_field[:, end])
        end
        # KNOWN GAP (documented, not fixed here): the VJP path builds its
        # initial state from cfg only — the field ICs T0f/W0f/C0f are NOT
        # threaded through, so gradients taken from a mid-cook state
        # (receding-horizon MPC) silently use uniform ICs. Whole-horizon
        # optimization and calibration are unaffected. t_wrap IS threaded.
        function _meat_vjp(θ, knots, tknots, tsave, abstol, reltol, t0,
                           t_wrap, cTs, cTc, cj, cq, cC, cW)
            cfg = MeatConfig(T0=t0)
            cot = (Tsurf=cTs, Tcore=cTc, j_evap=cj, q_meat=cq, C_field_final=cC,
                   w_retained=cW)
            g = meat_vjp(θ, knots, tknots, tsave, cot; cfg=cfg, abstol=abstol,
                         reltol=reltol, t_wrap=t_wrap)
            return g.dθ, g.dTpit_knots, g.dt_wrap
        end
        """
    )


with filelock.FileLock(Path(tempfile.gettempdir()) / "brisket_julia_init.lock", timeout=600):
    _on_julia_thread(_init)

NX = 40  # spatial nodes (MeatConfig default; c_field_final length)


def _jvec(x) -> "juliacall.VectorValue":
    return juliacall.convert(jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64)))


class InputSchema(BaseModel):
    theta: Differentiable[Array[(5,), Float64]] = Field(
        description="Meat parameters [D_eff m²/s, k_m m/s, h_c W/m²K, L m, k_mult]"
    )
    tpit_knots: Differentiable[Array[(None,), Float64]] = Field(
        description="Pit temperature trajectory [K] on the uniform knot grid `tknots`"
    )
    tknots: Array[(None,), Float64] = Field(
        description="Uniform time knot grid [s] for tpit_knots"
    )
    tsave: Array[(None,), Float64] = Field(description="Output save times [s]")
    abstol: float = Field(default=1e-6, description="ODE solver absolute tolerance")
    reltol: float = Field(default=1e-6, description="ODE solver relative tolerance")
    t0_meat: float = Field(default=281.15, description="Initial meat temperature [K]")
    t_wrap: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9,
        description="Texas-crutch wrap time [s]; default (1e9) = never wrap. "
                    "Smooth blend of width MeatConfig.δ_wrap; differentiable.")
    # Optional full-field initial state (receding-horizon state handoff);
    # empty arrays fall back to uniform ICs. Non-differentiable by design.
    t_field0: Array[(None,), Float64] = Field(
        default_factory=list, description="Initial T field (len nx) or empty")
    w_field0: Array[(None,), Float64] = Field(
        default_factory=list, description="Initial W field (len nx) or empty")
    c_field0: Array[(None,), Float64] = Field(
        default_factory=list, description="Initial C field (len nx) or empty")


class OutputSchema(BaseModel):
    tsurf: Differentiable[Array[(None,), Float64]] = Field(
        description="Surface temperature [K] at tsave"
    )
    tcore: Differentiable[Array[(None,), Float64]] = Field(
        description="Center (probe) temperature [K] at tsave"
    )
    j_evap: Differentiable[Array[(None,), Float64]] = Field(
        description="Evaporative mass flux [kg/m²/s] at tsave"
    )
    q_meat: Differentiable[Array[(None,), Float64]] = Field(
        description="Total heat drawn from the pit [W] at tsave (coupling output)"
    )
    c_field_final: Differentiable[Array[(NX,), Float64]] = Field(
        description="Final collagen conversion field (center → surface)"
    )
    w_retained: Differentiable[Array[(), Float64]] = Field(
        description="Juiciness proxy: mean retained moisture at pull / W0"
    )
    t_field_final: Array[(NX,), Float64] = Field(
        description="Final temperature field (state handoff; non-differentiable)")
    w_field_final: Array[(NX,), Float64] = Field(
        description="Final moisture field (state handoff; non-differentiable)")


# Soft-fail mode for MCMC: failed solves return sentinel outputs / zero
# gradients instead of raising, so pathological proposals become rejections.
# See pit tesseract_api.py for rationale.
SOFT_FAIL = os.environ.get("BRISKET_SOFT_FAIL", "0") == "1"
_SENTINEL_T = 1.0e4  # [K]


def _own(x) -> np.ndarray:
    """Eager owned copy of an input array, made ON THE CALLBACK THREAD.

    XLA guarantees callback input buffers only for the duration of the
    callback invocation. Our endpoints hop threads and hold arrays through
    packing, so without an immediate copy an input can be read after XLA
    recycled its buffer (observed: a valid damper schedule arriving as
    all-NaN, and a 73-knot trajectory arriving as 1-element).
    """
    return np.array(x, dtype=np.float64, copy=True)


def apply(inputs: InputSchema) -> OutputSchema:
    theta, tpit, tknots, tsave = (_own(inputs.theta), _own(inputs.tpit_knots),
                                  _own(inputs.tknots), _own(inputs.tsave))
    t0f, w0f, c0f = (_own(inputs.t_field0), _own(inputs.w_field0),
                     _own(inputs.c_field0))
    if tpit.shape != tknots.shape:
        raise ValueError(
            f"DEGENERATE MEAT INPUT: tpit_knots {tpit.shape} vs tknots {tknots.shape}")

    def _run():
        ts, tc, j, q, c, wr, tf, wf = jl._meat_apply(
            _jvec(theta),
            _jvec(tpit),
            _jvec(tknots),
            _jvec(tsave),
            float(inputs.abstol),
            float(inputs.reltol),
            float(inputs.t0_meat),
            _jvec(t0f),
            _jvec(w0f),
            _jvec(c0f),
            float(np.asarray(inputs.t_wrap)),
        )
        # .copy() is load-bearing: np.asarray over a Julia vector is a
        # zero-copy VIEW of Julia memory. If that view escapes this thread,
        # its eventual PyBuffer_Release calls back into Julia from the wrong
        # thread and can trigger a GC that waits forever for this (parked)
        # thread — a timing-dependent process-wide deadlock.
        return (np.asarray(ts).copy(), np.asarray(tc).copy(),
                np.asarray(j).copy(), np.asarray(q).copy(),
                np.asarray(c).copy(), np.float64(wr),
                np.asarray(tf).copy(), np.asarray(wf).copy())

    try:
        ts, tc, j, q, c, wr, tf, wf = _on_julia_thread(_run)
    except RuntimeError:
        if not SOFT_FAIL:
            raise
        nt = len(tsave)
        ts = tc = np.full(nt, _SENTINEL_T)
        j = q = np.zeros(nt)
        c = np.zeros(NX)
        wr = np.float64(0.0)
        tf = np.full(NX, _SENTINEL_T)
        wf = np.zeros(NX)
    return OutputSchema(tsurf=ts, tcore=tc, j_evap=j, q_meat=q, c_field_final=c,
                        w_retained=wr, t_field_final=tf, w_field_final=wf)


def abstract_eval(abstract_inputs):
    nt = abstract_inputs.tsave.shape[0]
    vec = lambda n: ShapeDType(shape=(n,), dtype="float64")
    return {
        "tsurf": vec(nt),
        "tcore": vec(nt),
        "j_evap": vec(nt),
        "q_meat": vec(nt),
        "c_field_final": vec(NX),
        "w_retained": ShapeDType(shape=(), dtype="float64"),
        "t_field_final": vec(NX),
        "w_field_final": vec(NX),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    theta, tpit, tknots, tsave = (_own(inputs.theta), _own(inputs.tpit_knots),
                                  _own(inputs.tknots), _own(inputs.tsave))
    nt = len(tsave)
    cots = {name: (_own(cotangent_vector[name])
                   if name in cotangent_vector and name in vjp_outputs else None)
            for name in ("tsurf", "tcore", "j_evap", "q_meat", "c_field_final",
                         "w_retained")}

    def _run():
        def cot(name, n):
            v = cots[name]
            return _jvec(v if v is not None else np.zeros(n))

        dtheta, dknots, dtwrap = jl._meat_vjp(
            _jvec(theta),
            _jvec(tpit),
            _jvec(tknots),
            _jvec(tsave),
            float(inputs.abstol),
            float(inputs.reltol),
            float(inputs.t0_meat),
            float(np.asarray(inputs.t_wrap)),
            cot("tsurf", nt),
            cot("tcore", nt),
            cot("j_evap", nt),
            cot("q_meat", nt),
            cot("c_field_final", NX),
            float(cots["w_retained"]) if cots["w_retained"] is not None else 0.0,
        )
        # .copy(): no zero-copy views of Julia memory may cross threads
        return (np.asarray(dtheta).copy(), np.asarray(dknots).copy(),
                float(dtwrap))

    try:
        dtheta, dknots, dtwrap = _on_julia_thread(_run)
    except RuntimeError:
        if not SOFT_FAIL:
            raise
        dtheta, dknots, dtwrap = np.zeros(len(theta)), np.zeros(len(tpit)), 0.0
    if np.isnan(dtheta).any() or np.isnan(dknots).any() or np.isnan(dtwrap):
        # NaN gradients propagate silently through the whole pipeline (they
        # reached the optimizer state once and cost us an evening) — capture
        # the exact call for offline replay in Julia, then fail loudly.
        cap = Path(tempfile.gettempdir()) / "meat_vjp_nan_capture.npz"
        np.savez(cap, theta=theta, tpit=tpit, tknots=tknots, tsave=tsave,
                 abstol=inputs.abstol, reltol=inputs.reltol,
                 **{f"cot_{k}": v for k, v in cots.items() if v is not None})
        raise FloatingPointError(
            f"meat VJP produced NaN gradients; failing call captured to {cap}")
    out = {}
    if "theta" in vjp_inputs:
        out["theta"] = dtheta
    if "tpit_knots" in vjp_inputs:
        out["tpit_knots"] = dknots
    if "t_wrap" in vjp_inputs:
        out["t_wrap"] = np.asarray(dtwrap, dtype=np.float64)
    return out
