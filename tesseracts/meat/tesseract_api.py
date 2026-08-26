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
    _libs = _glob.glob(os.path.join(_libdir, "libpython*.so.1.0")) or _glob.glob(
        os.path.join(_libdir, "libpython*.so")
    )
    if _libs:
        os.environ["JULIA_PYTHONCALL_LIB"] = _libs[0]

import filelock

_EXEC_ATTR = "_brisket_julia_executor"
if getattr(sys, _EXEC_ATTR, None) is None:
    setattr(
        sys,
        _EXEC_ATTR,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="juliacall"),
    )
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
        # Blind catch is intentional: ANY exception object originating on
        # the Julia thread must be stringified here, not propagated.
        except Exception as e:  # noqa: BLE001
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
                             T0f, W0f, C0f, F0f, b0, s0, t_wrap, smoke)
            cfg = MeatConfig(T0=t0)
            o = meat_solve(θ, knots, tknots, tsave; cfg=cfg, abstol=abstol,
                           reltol=reltol, T0f=T0f, W0f=W0f, C0f=C0f,
                           F0f=F0f, B0=b0, S0=s0,
                           t_wrap=t_wrap, smoke_knots=smoke)
            return (o.Tsurf, o.Tcore, o.j_evap, o.q_meat, o.C_field_final,
                    o.w_retained, o.T_field[:, end], o.W_field[:, end],
                    o.bark_b, o.bark_s, o.l_star,
                    o.juiciness, o.fat_rendered, o.evap_w, o.drip_w, o.drip_f,
                    o.F_field[:, end])
        end
        # The VJP threads the full field ICs (T0f/W0f/C0f/F0f/B0/S0) so the
        # adjoint problem starts from the SAME state as the primal —
        # receding-horizon (mid-cook) gradients differentiate their own
        # forecast, not a fresh default brisket.
        function _meat_vjp(θ, knots, tknots, tsave, abstol, reltol, t0,
                           T0f, W0f, C0f, F0f, b0, s0,
                           t_wrap, smoke, cTs, cTc, cj, cq, cC, cW, cB, cS, cL,
                           cJu, cFr)
            cfg = MeatConfig(T0=t0)
            cot = (Tsurf=cTs, Tcore=cTc, j_evap=cj, q_meat=cq, C_field_final=cC,
                   w_retained=cW, bark_b=cB, bark_s=cS, l_star=cL,
                   juiciness=cJu, fat_rendered=cFr)
            g = meat_vjp(θ, knots, tknots, tsave, cot; cfg=cfg, abstol=abstol,
                         reltol=reltol, t_wrap=t_wrap, smoke_knots=smoke,
                         T0f=T0f, W0f=W0f, C0f=C0f, F0f=F0f, B0=b0, S0=s0)
            return g.dθ, g.dTpit_knots, g.dsmoke_knots, g.dt_wrap
        end
        """
    )


with filelock.FileLock(
    Path(tempfile.gettempdir()) / "brisket_julia_init.lock", timeout=600
):
    _on_julia_thread(_init)

NX = 40  # spatial nodes (MeatConfig default; c_field_final length)


def _jvec(x) -> "juliacall.VectorValue":
    return juliacall.convert(
        jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    )


class InputSchema(BaseModel):
    """Inputs to the meat solve.

    Differentiable: the meat parameter vector ``theta``, the pit-temperature
    and smoke forcings on the ``tknots`` grid, and the wrap time ``t_wrap``.
    Non-differentiable: time grids, solver tolerances, the scalar initial
    temperature, and the optional full-field initial state used for
    receding-horizon state handoff.
    """

    theta: Differentiable[Array[(7,), Float64]] = Field(
        description="Meat parameters [D_eff m²/s, k_m m/s, h_c W/m²K, L m, "
        "k_mult, k_e 1/s, a_myosin]"
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
        "Smooth blend of width MeatConfig.δ_wrap; differentiable.",
    )
    smoke_knots: Differentiable[Array[(None,), Float64]] = Field(
        default_factory=list,
        description="Smoke-flux forcing (pit burn rate, kg/s) on the tknots "
        "grid; empty = no smoke. Drives the bark deposition state; "
        "differentiable so gradients flow damper -> burn rate -> "
        "smoke -> bark.",
    )
    # Optional full-field initial state (receding-horizon state handoff);
    # empty arrays fall back to uniform ICs. Non-differentiable by design.
    t_field0: Array[(None,), Float64] = Field(
        default_factory=list, description="Initial T field (len nx) or empty"
    )
    w_field0: Array[(None,), Float64] = Field(
        default_factory=list, description="Initial W field (len nx) or empty"
    )
    c_field0: Array[(None,), Float64] = Field(
        default_factory=list, description="Initial C field (len nx) or empty"
    )
    f_field0: Array[(None,), Float64] = Field(
        default_factory=list,
        description="Initial un-rendered fat field (len nx) or empty. Part "
        "of the full state handoff: without it a replanned segment "
        "re-renders fat from scratch and mis-scores juiciness.",
    )
    bark_b0: Float64 = Field(
        default=float("nan"),
        description="Initial Maillard bark state (NaN = 0; state handoff)",
    )
    bark_s0: Float64 = Field(
        default=float("nan"),
        description="Initial smoke-deposition state (NaN = 0; state handoff)",
    )


class OutputSchema(BaseModel):
    """Outputs of the meat solve.

    Time series at ``tsave`` (temperatures, evaporative flux, pit heat
    load, bark states, L*), scalar quality summaries, mass-budget
    diagnostics, and the final T/W fields for receding-horizon state
    handoff. Only the fields typed ``Differentiable`` participate in the
    VJP endpoint; the diagnostics and final fields do not.
    """

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
    bark_b: Differentiable[Array[(None,), Float64]] = Field(
        description="Bark Maillard browning state B in [0,1) at tsave"
    )
    bark_s: Differentiable[Array[(None,), Float64]] = Field(
        description="Bark smoke-deposition state S in [0,1) at tsave"
    )
    l_star: Differentiable[Array[(None,), Float64]] = Field(
        description="Predicted surface lightness L* (CIELAB) at tsave"
    )
    juiciness: Differentiable[Array[(), Float64]] = Field(
        description="Perceptual juiciness: retained water + β·retained "
        "rendered fat, as water-equivalent fraction of W0"
    )
    fat_rendered: Differentiable[Array[(), Float64]] = Field(
        description="Mean rendered fat fraction at pull (0-1)"
    )
    evap_w: Array[(), Float64] = Field(
        description="Evaporated water, dry-basis mean units (scale calib)"
    )
    drip_w: Array[(), Float64] = Field(
        description="Dripped (purged) water, dry-basis mean units"
    )
    drip_f: Array[(), Float64] = Field(
        description="Drained rendered fat (tallow catch), dry-basis mean units"
    )
    t_field_final: Array[(NX,), Float64] = Field(
        description="Final temperature field (state handoff; non-differentiable)"
    )
    w_field_final: Array[(NX,), Float64] = Field(
        description="Final moisture field (state handoff; non-differentiable)"
    )
    f_field_final: Array[(NX,), Float64] = Field(
        description="Final un-rendered fat field (state handoff; "
        "non-differentiable)"
    )


# Soft-fail mode for MCMC: failed solves return sentinel outputs / zero
# gradients instead of raising, so pathological proposals become rejections.
# See pit tesseract_api.py for rationale.
SOFT_FAIL = os.environ.get("BRISKET_SOFT_FAIL", "0") == "1"
_SENTINEL_T = 1.0e4  # [K]
_SOFT_FAIL_COUNT = 0
_SOLVER_FAIL_MARKERS = (
    "solve failed",
    "retcode",
    "nan inputs",
    "dt_epsilon",
    "identically zero gradients",
    "backward solve",
    "dt < eps",
)


def _is_solver_failure(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _SOLVER_FAIL_MARKERS)


def _maybe_soft_fail(err: RuntimeError, what: str) -> None:
    """Re-raise programming errors; swallow classified solver failures when enabled."""
    global _SOFT_FAIL_COUNT
    if not SOFT_FAIL or not _is_solver_failure(err):
        raise
    _SOFT_FAIL_COUNT += 1
    if _SOFT_FAIL_COUNT == 1 or _SOFT_FAIL_COUNT % 10 == 0:
        # Windows cp1252 consoles cannot print Julia's θ in the error text;
        # a UnicodeEncodeError here would abort NUTS instead of rejecting.
        msg = str(err).encode("ascii", "replace").decode("ascii")
        print(f"SOFT_FAIL {what} #{_SOFT_FAIL_COUNT}: {msg}", flush=True)


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
    """Solve the 1D heat-moisture brisket PDE (Tesseract apply endpoint).

    Copies every input buffer on the callback thread (XLA only guarantees
    them for the duration of the invocation), validates that the pit
    trajectory matches its knot grid, and runs ``meat_solve`` on the
    dedicated Julia thread.

    Parameters
    ----------
    inputs : InputSchema
        Meat parameters, pit/smoke forcings with their knot grid, save
        times, tolerances, wrap time, and optional initial fields.

    Returns
    -------
    OutputSchema
        Surface/core temperatures, evaporative flux, pit heat load, bark
        states and L*, quality scalars, mass-budget diagnostics, and the
        final T/W fields. With BRISKET_SOFT_FAIL=1 a failed solve returns
        sentinel outputs (1e4 K temperatures, zeroed states) instead of
        raising, so MCMC treats pathological proposals as rejections.

    Raises
    ------
    ValueError
        If ``tpit_knots`` and ``tknots`` have different shapes (degenerate
        input).
    RuntimeError
        If the Julia solve fails and soft-fail mode is off.
    """
    theta, tpit, tknots, tsave = (
        _own(inputs.theta),
        _own(inputs.tpit_knots),
        _own(inputs.tknots),
        _own(inputs.tsave),
    )
    t0f, w0f, c0f, f0f = (
        _own(inputs.t_field0),
        _own(inputs.w_field0),
        _own(inputs.c_field0),
        _own(inputs.f_field0),
    )
    smoke = _own(inputs.smoke_knots)
    if tpit.shape != tknots.shape:
        raise ValueError(
            f"DEGENERATE MEAT INPUT: tpit_knots {tpit.shape} vs tknots {tknots.shape}"
        )

    def _run():
        (ts, tc, j, q, c, wr, tf, wf, bb, bs, ls, ju, fr, ew, dw, df, ff) = (
            jl._meat_apply(
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
                _jvec(f0f),
                float(inputs.bark_b0),
                float(inputs.bark_s0),
                float(np.asarray(inputs.t_wrap)),
                _jvec(smoke),
            )
        )
        # .copy() is load-bearing: np.asarray over a Julia vector is a
        # zero-copy VIEW of Julia memory. If that view escapes this thread,
        # its eventual PyBuffer_Release calls back into Julia from the wrong
        # thread and can trigger a GC that waits forever for this (parked)
        # thread — a timing-dependent process-wide deadlock.
        return (
            np.asarray(ts).copy(),
            np.asarray(tc).copy(),
            np.asarray(j).copy(),
            np.asarray(q).copy(),
            np.asarray(c).copy(),
            np.float64(wr),
            np.asarray(tf).copy(),
            np.asarray(wf).copy(),
            np.asarray(bb).copy(),
            np.asarray(bs).copy(),
            np.asarray(ls).copy(),
            np.float64(ju),
            np.float64(fr),
            np.float64(ew),
            np.float64(dw),
            np.float64(df),
            np.asarray(ff).copy(),
        )

    try:
        (ts, tc, j, q, c, wr, tf, wf, bb, bs, ls, ju, fr, ew, dw, df, ff) = (
            _on_julia_thread(_run)
        )
    except RuntimeError as e:
        _maybe_soft_fail(e, "meat apply")
        nt = len(tsave)
        ts = tc = np.full(nt, _SENTINEL_T)
        j = q = np.zeros(nt)
        c = np.zeros(NX)
        wr = np.float64(0.0)
        tf = np.full(NX, _SENTINEL_T)
        wf = np.zeros(NX)
        bb = bs = np.zeros(nt)
        ls = np.full(nt, 52.0)
        ju = fr = ew = dw = df = np.float64(0.0)
        ff = np.zeros(NX)
    return OutputSchema(
        tsurf=ts,
        tcore=tc,
        j_evap=j,
        q_meat=q,
        c_field_final=c,
        w_retained=wr,
        t_field_final=tf,
        w_field_final=wf,
        f_field_final=ff,
        bark_b=bb,
        bark_s=bs,
        l_star=ls,
        juiciness=ju,
        fat_rendered=fr,
        evap_w=ew,
        drip_w=dw,
        drip_f=df,
    )


def abstract_eval(abstract_inputs):
    """Report output shapes/dtypes without solving (abstract_eval endpoint).

    Parameters
    ----------
    abstract_inputs
        Abstract input pytree; only ``tsave.shape[0]`` is consulted.

    Returns
    -------
    dict of str to ShapeDType
        One entry per OutputSchema field: float64 time series of length
        ``len(tsave)``, fields of length NX, and float64 scalars.
    """
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
        "f_field_final": vec(NX),
        "bark_b": vec(nt),
        "bark_s": vec(nt),
        "l_star": vec(nt),
        "juiciness": ShapeDType(shape=(), dtype="float64"),
        "fat_rendered": ShapeDType(shape=(), dtype="float64"),
        "evap_w": ShapeDType(shape=(), dtype="float64"),
        "drip_w": ShapeDType(shape=(), dtype="float64"),
        "drip_f": ShapeDType(shape=(), dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Vector-Jacobian product of the meat solve (Tesseract VJP endpoint).

    Backed by the SciMLSensitivity continuous adjoint of the stiff
    method-of-lines solve, run on the dedicated Julia thread. Missing
    cotangents are treated as zeros. The optional field ICs
    (t_field0/w_field0/c_field0/f_field0/bark_b0/bark_s0) are threaded
    through, so the adjoint problem starts from the same state as the
    primal (mid-cook / receding-horizon gradients included).

    Parameters
    ----------
    inputs : InputSchema
        The primal inputs of the corresponding apply call.
    vjp_inputs : set of str
        Differentiable input names to return gradients for (subset of
        {"theta", "tpit_knots", "t_wrap", "smoke_knots"}).
    vjp_outputs : set of str
        Output names whose cotangents in ``cotangent_vector`` are live.
    cotangent_vector : dict
        Cotangent arrays keyed by differentiable output name.

    Returns
    -------
    dict of str to numpy.ndarray
        Gradients for the requested inputs. With BRISKET_SOFT_FAIL=1 a
        failed adjoint solve returns zeros (the proposal is being rejected
        anyway; zeros keep HMC finite).

    Raises
    ------
    FloatingPointError
        If any returned gradient contains NaN: the exact failing call is
        captured to ``meat_vjp_nan_capture.npz`` in the temp dir for
        offline replay in Julia, then the endpoint fails loudly rather
        than let NaNs propagate silently into the optimizer.
    RuntimeError
        If the Julia adjoint solve fails and soft-fail mode is off.
    """
    theta, tpit, tknots, tsave = (
        _own(inputs.theta),
        _own(inputs.tpit_knots),
        _own(inputs.tknots),
        _own(inputs.tsave),
    )
    t0f, w0f, c0f, f0f = (
        _own(inputs.t_field0),
        _own(inputs.w_field0),
        _own(inputs.c_field0),
        _own(inputs.f_field0),
    )
    nt = len(tsave)
    smoke = _own(inputs.smoke_knots)
    cots = {
        name: (
            _own(cotangent_vector[name])
            if name in cotangent_vector and name in vjp_outputs
            else None
        )
        for name in (
            "tsurf",
            "tcore",
            "j_evap",
            "q_meat",
            "c_field_final",
            "w_retained",
            "bark_b",
            "bark_s",
            "l_star",
            "juiciness",
            "fat_rendered",
        )
    }

    def _run():
        def cot(name, n):
            v = cots[name]
            return _jvec(v if v is not None else np.zeros(n))

        dtheta, dknots, dsmoke, dtwrap = jl._meat_vjp(
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
            _jvec(f0f),
            float(inputs.bark_b0),
            float(inputs.bark_s0),
            float(np.asarray(inputs.t_wrap)),
            _jvec(smoke),
            cot("tsurf", nt),
            cot("tcore", nt),
            cot("j_evap", nt),
            cot("q_meat", nt),
            cot("c_field_final", NX),
            float(cots["w_retained"]) if cots["w_retained"] is not None else 0.0,
            cot("bark_b", nt),
            cot("bark_s", nt),
            cot("l_star", nt),
            float(cots["juiciness"]) if cots["juiciness"] is not None else 0.0,
            float(cots["fat_rendered"]) if cots["fat_rendered"] is not None else 0.0,
        )
        # .copy(): no zero-copy views of Julia memory may cross threads
        return (
            np.asarray(dtheta).copy(),
            np.asarray(dknots).copy(),
            np.asarray(dsmoke).copy(),
            float(dtwrap),
        )

    try:
        dtheta, dknots, dsmoke, dtwrap = _on_julia_thread(_run)
    except RuntimeError as e:
        _maybe_soft_fail(e, "meat vjp")
        dtheta, dknots, dsmoke, dtwrap = (
            np.zeros(len(theta)),
            np.zeros(len(tpit)),
            np.zeros(len(smoke)),
            0.0,
        )
    if (
        np.isnan(dtheta).any()
        or np.isnan(dknots).any()
        or np.isnan(dsmoke).any()
        or np.isnan(dtwrap)
    ):
        # NaN gradients propagate silently through the whole pipeline (they
        # reached the optimizer state once and cost us an evening) — capture
        # the exact call for offline replay in Julia, then fail loudly.
        cap = Path(tempfile.gettempdir()) / "meat_vjp_nan_capture.npz"
        np.savez(
            cap,
            theta=theta,
            tpit=tpit,
            tknots=tknots,
            tsave=tsave,
            abstol=inputs.abstol,
            reltol=inputs.reltol,
            **{f"cot_{k}": v for k, v in cots.items() if v is not None},
        )
        raise FloatingPointError(
            f"meat VJP produced NaN gradients; failing call captured to {cap}"
        )
    out = {}
    if "theta" in vjp_inputs:
        out["theta"] = dtheta
    if "tpit_knots" in vjp_inputs:
        out["tpit_knots"] = dknots
    if "t_wrap" in vjp_inputs:
        out["t_wrap"] = np.asarray(dtwrap, dtype=np.float64)
    if "smoke_knots" in vjp_inputs:
        out["smoke_knots"] = dsmoke
    return out
