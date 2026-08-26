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
    _libs = _glob.glob(os.path.join(_libdir, "libpython*.so.1.0")) or _glob.glob(
        os.path.join(_libdir, "libpython*.so")
    )
    if _libs:
        os.environ["JULIA_PYTHONCALL_LIB"] = _libs[0]

import filelock

# All juliacall work runs on one dedicated thread, shared process-wide with the
# meat Tesseract (sys-level singleton) — jitted JAX callbacks arrive on XLA
# worker threads, and juliacall off its init thread deadlocks. See the meat
# tesseract_api.py docstring for the full story.
_EXEC_ATTR = "_brisket_julia_executor"
if getattr(sys, _EXEC_ATTR, None) is None:
    setattr(
        sys,
        _EXEC_ATTR,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="juliacall"),
    )
_JL_THREAD: ThreadPoolExecutor = getattr(sys, _EXEC_ATTR)


# Direct mode for the containerized runtime — see meat tesseract_api.py.
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
            return o.Tpit, o.m_fuel, o.ignited, o.mdot
        end
        function _pit_vjp(φ, u_knots, u_tknots, q_knots, q_tknots, tsave,
                          abstol, reltol, tpit0, t_amb, m0, cT, cm, cmd)
            cfg = PitConfig(Tpit0=tpit0, T_amb=t_amb, m0=m0)
            g = pit_vjp(φ, u_knots, u_tknots, q_knots, q_tknots, tsave,
                        (Tpit=cT, m_fuel=cm, mdot=cmd); cfg=cfg, abstol=abstol,
                        reltol=reltol)
            return g.dφ, g.du_knots, g.dqmeat_knots
        end
        """
    )


with filelock.FileLock(
    Path(tempfile.gettempdir()) / "brisket_julia_init.lock", timeout=600
):
    _on_julia_thread(_init)


def _jvec(x) -> "juliacall.VectorValue":
    return juliacall.convert(
        jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    )


class InputSchema(BaseModel):
    """Inputs to the pit solve.

    Differentiable: the pit parameter vector ``phi``, the damper schedule,
    and the meat heat-load forcing. Non-differentiable: the knot grids,
    save times, solver tolerances, and the scalar initial conditions
    (initial pit temperature, ambient temperature, fuel load).
    """

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
    """Outputs of the pit solve, all evaluated at ``tsave``.

    Pit temperature, remaining fuel mass, and the burn rate (the smoke-flux
    proxy) are differentiable; the ignited bed fraction is a
    diagnostic/state-handoff output outside the VJP.
    """

    tpit: Differentiable[Array[(None,), Float64]] = Field(
        description="Pit gas temperature [K] at tsave"
    )
    m_fuel: Differentiable[Array[(None,), Float64]] = Field(
        description="Remaining fuel mass [kg] at tsave"
    )
    ignited: Array[(None,), Float64] = Field(
        description="Ignited bed fraction at tsave (diagnostic/state handoff)"
    )
    mdot: Differentiable[Array[(None,), Float64]] = Field(
        description="Exact instantaneous fuel burn rate [kg/s] at tsave "
        "(smoke-flux proxy for the bark model)"
    )


def _own(x) -> np.ndarray:
    """Eager owned copy of an input array, made ON THE CALLBACK THREAD.

    XLA only guarantees callback input buffers during the invocation; see
    meat tesseract_api.py.
    """
    return np.array(x, dtype=np.float64, copy=True)


# Soft-fail mode (BRISKET_SOFT_FAIL=1): a failed solve returns sentinel
# outputs (absurd temperatures -> astronomically bad likelihood) instead of
# raising, so MCMC treats pathological parameter proposals as rejections
# rather than crashing the chain. NUTS warmup legitimately probes regions
# where the pit ODE is unsolvable (e.g. tiny C_th); everything outside
# sampling keeps hard failures.
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


def apply(inputs: InputSchema) -> OutputSchema:
    """Solve the lumped-parameter smoker ODE (Tesseract apply endpoint).

    Copies every input buffer on the callback thread (XLA only guarantees
    them for the duration of the invocation) and runs ``pit_solve`` on the
    dedicated Julia thread.

    Parameters
    ----------
    inputs : InputSchema
        Pit parameters, damper and heat-load forcings with their knot
        grids, save times, tolerances, and initial conditions.

    Returns
    -------
    OutputSchema
        Pit temperature, remaining fuel, ignited bed fraction, and burn
        rate at ``tsave``. With BRISKET_SOFT_FAIL=1 a failed solve returns
        sentinel outputs (1e4 K temperatures -> astronomically bad
        likelihood) instead of raising, so MCMC treats pathological
        proposals as rejections.

    Raises
    ------
    RuntimeError
        If the Julia solve fails and soft-fail mode is off.
    """
    phi, u, utk, q, qtk, tsave = (
        _own(inputs.phi),
        _own(inputs.u_knots),
        _own(inputs.u_tknots),
        _own(inputs.qmeat_knots),
        _own(inputs.q_tknots),
        _own(inputs.tsave),
    )
    if u.shape != utk.shape:
        raise ValueError(
            f"DEGENERATE PIT INPUT: u_knots {u.shape} vs u_tknots {utk.shape}"
        )
    if q.shape != qtk.shape:
        raise ValueError(
            f"DEGENERATE PIT INPUT: qmeat_knots {q.shape} vs q_tknots {qtk.shape}"
        )

    def _run():
        T, m, a, md = jl._pit_apply(
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
        return (
            np.asarray(T).copy(),
            np.asarray(m).copy(),
            np.asarray(a).copy(),
            np.asarray(md).copy(),
        )

    try:
        T, m, a, md = _on_julia_thread(_run)
    except RuntimeError as e:
        _maybe_soft_fail(e, "pit apply")
        nt = len(tsave)
        T, m, a, md = (
            np.full(nt, _SENTINEL_T),
            np.zeros(nt),
            np.zeros(nt),
            np.zeros(nt),
        )
    return OutputSchema(tpit=T, m_fuel=m, ignited=a, mdot=md)


def abstract_eval(abstract_inputs):
    """Report output shapes/dtypes without solving (abstract_eval endpoint).

    Parameters
    ----------
    abstract_inputs
        Abstract input pytree; only ``tsave.shape[0]`` is consulted.

    Returns
    -------
    dict of str to ShapeDType
        Four float64 vectors of length ``len(tsave)``: tpit, m_fuel,
        ignited, mdot.
    """
    nt = abstract_inputs.tsave.shape[0]
    return {
        "tpit": ShapeDType(shape=(nt,), dtype="float64"),
        "m_fuel": ShapeDType(shape=(nt,), dtype="float64"),
        "ignited": ShapeDType(shape=(nt,), dtype="float64"),
        "mdot": ShapeDType(shape=(nt,), dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Vector-Jacobian product of the pit solve (Tesseract VJP endpoint).

    Computes a chunked ForwardDiff dense Jacobian in Julia and returns
    Jᵀv̄ — cheap and exact for the small non-stiff system, and deliberately
    a different differentiation strategy from the meat Tesseract's
    continuous adjoint. Missing cotangents are treated as zeros.

    Parameters
    ----------
    inputs : InputSchema
        The primal inputs of the corresponding apply call.
    vjp_inputs : set of str
        Differentiable input names to return gradients for (subset of
        {"phi", "u_knots", "qmeat_knots"}).
    vjp_outputs : set of str
        Output names whose cotangents in ``cotangent_vector`` are live
        (subset of {"tpit", "m_fuel", "mdot"}).
    cotangent_vector : dict
        Cotangent arrays keyed by differentiable output name.

    Returns
    -------
    dict of str to numpy.ndarray
        Gradients for the requested inputs. With BRISKET_SOFT_FAIL=1 a
        failed solve returns zeros: the sampler is rejecting the proposal
        anyway (sentinel likelihood), and zeros keep HMC finite.

    Raises
    ------
    RuntimeError
        If the Julia call fails and soft-fail mode is off.
    """
    phi, u, utk, q, qtk, tsave = (
        _own(inputs.phi),
        _own(inputs.u_knots),
        _own(inputs.u_tknots),
        _own(inputs.qmeat_knots),
        _own(inputs.q_tknots),
        _own(inputs.tsave),
    )
    if u.shape != utk.shape:
        raise ValueError(
            f"DEGENERATE PIT INPUT: u_knots {u.shape} vs u_tknots {utk.shape}"
        )
    if q.shape != qtk.shape:
        raise ValueError(
            f"DEGENERATE PIT INPUT: qmeat_knots {q.shape} vs q_tknots {qtk.shape}"
        )
    nt = len(tsave)
    cots = {
        name: (
            _own(cotangent_vector[name])
            if name in cotangent_vector and name in vjp_outputs
            else None
        )
        for name in ("tpit", "m_fuel", "mdot")
    }

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
            cot("mdot"),
        )
        # .copy(): no zero-copy views of Julia memory may cross threads
        return (np.asarray(dphi).copy(), np.asarray(du).copy(), np.asarray(dq).copy())

    try:
        dphi, du, dq = _on_julia_thread(_run)
    except RuntimeError as e:
        _maybe_soft_fail(e, "pit vjp")
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
