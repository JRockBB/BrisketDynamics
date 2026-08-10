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

REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

import filelock

_EXEC_ATTR = "_brisket_julia_executor"
if getattr(sys, _EXEC_ATTR, None) is None:
    setattr(sys, _EXEC_ATTR, ThreadPoolExecutor(max_workers=1, thread_name_prefix="juliacall"))
_JL_THREAD: ThreadPoolExecutor = getattr(sys, _EXEC_ATTR)


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

    return _JL_THREAD.submit(_guarded).result()


def _init():
    global juliacall, jl
    import juliacall

    jl = juliacall.newmodule("MeatTess")
    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO_ROOT}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jl.seval(
        """
        function _meat_apply(θ, knots, tknots, tsave, abstol, reltol)
            o = meat_solve(θ, knots, tknots, tsave; abstol=abstol, reltol=reltol)
            return o.Tsurf, o.Tcore, o.j_evap, o.q_meat, o.C_field_final
        end
        function _meat_vjp(θ, knots, tknots, tsave, abstol, reltol,
                           cTs, cTc, cj, cq, cC)
            cot = (Tsurf=cTs, Tcore=cTc, j_evap=cj, q_meat=cq, C_field_final=cC)
            g = meat_vjp(θ, knots, tknots, tsave, cot; abstol=abstol, reltol=reltol)
            return g.dθ, g.dTpit_knots
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
    if tpit.shape != tknots.shape:
        raise ValueError(
            f"DEGENERATE MEAT INPUT: tpit_knots {tpit.shape} vs tknots {tknots.shape}")

    def _run():
        ts, tc, j, q, c = jl._meat_apply(
            _jvec(theta),
            _jvec(tpit),
            _jvec(tknots),
            _jvec(tsave),
            float(inputs.abstol),
            float(inputs.reltol),
        )
        # .copy() is load-bearing: np.asarray over a Julia vector is a
        # zero-copy VIEW of Julia memory. If that view escapes this thread,
        # its eventual PyBuffer_Release calls back into Julia from the wrong
        # thread and can trigger a GC that waits forever for this (parked)
        # thread — a timing-dependent process-wide deadlock.
        return (np.asarray(ts).copy(), np.asarray(tc).copy(),
                np.asarray(j).copy(), np.asarray(q).copy(),
                np.asarray(c).copy())

    ts, tc, j, q, c = _on_julia_thread(_run)
    return OutputSchema(tsurf=ts, tcore=tc, j_evap=j, q_meat=q, c_field_final=c)


def abstract_eval(abstract_inputs):
    nt = abstract_inputs.tsave.shape[0]
    vec = lambda n: ShapeDType(shape=(n,), dtype="float64")
    return {
        "tsurf": vec(nt),
        "tcore": vec(nt),
        "j_evap": vec(nt),
        "q_meat": vec(nt),
        "c_field_final": vec(NX),
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
            for name in ("tsurf", "tcore", "j_evap", "q_meat", "c_field_final")}

    def _run():
        def cot(name, n):
            v = cots[name]
            return _jvec(v if v is not None else np.zeros(n))

        dtheta, dknots = jl._meat_vjp(
            _jvec(theta),
            _jvec(tpit),
            _jvec(tknots),
            _jvec(tsave),
            float(inputs.abstol),
            float(inputs.reltol),
            cot("tsurf", nt),
            cot("tcore", nt),
            cot("j_evap", nt),
            cot("q_meat", nt),
            cot("c_field_final", NX),
        )
        # .copy(): no zero-copy views of Julia memory may cross threads
        return np.asarray(dtheta).copy(), np.asarray(dknots).copy()

    dtheta, dknots = _on_julia_thread(_run)
    if np.isnan(dtheta).any() or np.isnan(dknots).any():
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
    return out
