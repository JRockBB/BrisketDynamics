"""Monolith Tesseract: the whole coupled cook (pit + meat + humidity) as one endpoint.

Wraps BrisketDynamics.jl's `monolith_solve` and its forward-mode
derivatives. One stiff solve for the coupled system and forward-mode
(ForwardDiff) derivatives: the
parameter count is small and the forward solve is ~100x cheaper than a
continuous adjoint, so `jacobian_vector_product` is one Dual solve,
`vector_jacobian_product` and `jacobian` are one chunked Jacobian pass over
the requested inputs. All three are exposed, so JAX gets reverse AND
forward mode through tesseract-jax, and Gauss-Newton / Fisher / Newton
control get the full Jacobian.

Threading follows the meat Tesseract: every juliacall call runs on the one
process-wide Julia thread (`sys._brisket_julia_executor`).
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
    REPO_ROOT = Path("/app/BrisketDynamics")

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

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
_DIRECT = os.environ.get("BRISKET_JL_DIRECT", "0") == "1"


def _on_julia_thread(fn, *args):
    def _guarded():
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 — stringify ON the Julia thread
            raise RuntimeError(f"Julia call failed: {type(e).__name__}: {e}") from None

    if _DIRECT:
        return _guarded()
    return _JL_THREAD.submit(_guarded).result()


def _init():
    global juliacall, jl
    import juliacall

    jl = juliacall.newmodule("MonoTess")
    _project = os.environ.get("JULIA_PROJECT") or str(REPO_ROOT)
    jl.seval(f'import Pkg; Pkg.activate(raw"{_project}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jl.seval(
        """
        function _mono_kw(abstol, reltol, tpit0, t_amb, m0, t0, T0f, W0f, C0f, F0f,
                          b0, s0, r0, rest_t, rest_h, rest_eps, rest_eta, n_shrink,
                          afr, y_w, pan, omega, n_meat)
            cfg = MeatConfig(T0=t0, T_rest=rest_t, h_rest=rest_h, ε_rest=rest_eps,
                             η_rest=rest_eta, n_shrink=n_shrink, n_meat=n_meat)
            pit = PitConfig(T_amb=t_amb, Tpit0=tpit0, m0=m0)
            hum = Humidity(afr=afr, y_w=y_w, pan_kg_s=pan, omega_amb=omega)
            return (; cfg, pit, hum, abstol, reltol, T0f, W0f, C0f, F0f, B0=b0, S0=s0,
                    R0=r0)
        end
        function _mono_apply(θ, φ, u, utk, tsave, t_wrap, t_pull, t_add, t_spritz,
                             dt_spritz, m_spritz, kwargs...)
            kw = _mono_kw(kwargs...)
            o = monolith_solve(θ, φ, u, utk, tsave; t_wrap, t_pull, t_add, t_spritz,
                               dt_spritz, m_spritz, kw...)
            return (o.Tpit, o.Tcore, o.Tsurf, o.m_fuel, o.ignited, o.mdot, o.j_evap,
                    o.q_meat, o.C_field_final, o.w_retained, o.bark_b, o.bark_s,
                    o.l_star, o.juiciness, o.fat_rendered, o.l_thick, o.pv_pit,
                    o.pv_base, o.evap_w, o.drip_w, o.drip_f, o.T_field[:, end],
                    o.W_field[:, end], o.F_field[:, end], o.ring[end], o.spritz_w)
        end
        function _mono_jac(θ, φ, u, utk, tsave, t_wrap, t_pull, t_add, t_spritz,
                           dt_spritz, m_spritz, sel, kwargs...)
            kw = _mono_kw(kwargs...)
            return monolith_stack_jacobian(θ, φ, u, utk, tsave, t_wrap, t_pull, t_add,
                                           t_spritz, dt_spritz, m_spritz,
                                           collect(Int, sel); kw...)
        end
        function _mono_jvp(θ, φ, u, utk, tsave, t_wrap, t_pull, t_add, t_spritz,
                           dt_spritz, m_spritz, dv, kwargs...)
            kw = _mono_kw(kwargs...)
            return monolith_stack_jvp(θ, φ, u, utk, tsave, t_wrap, t_pull, t_add,
                                      t_spritz, dt_spritz, m_spritz,
                                      collect(Float64, dv); kw...)
        end
        """
    )


with filelock.FileLock(
    Path(tempfile.gettempdir()) / "brisket_julia_init.lock", timeout=600
):
    _on_julia_thread(_init)

NX = 40  # spatial nodes (MeatConfig default)
N_THETA, N_PHI = 8, 7
# the stacked differentiable outputs, in monolith_stack order
STACK_SERIES = (
    "tpit",
    "tcore",
    "tsurf",
    "m_fuel",
    "j_evap",
    "q_meat",
    "bark_b",
    "bark_s",
    "l_star",
    "l_thick",
)
STACK_FIELDS = ("c_field_final",)
STACK_SCALARS = ("w_retained", "juiciness", "fat_rendered")
DIFF_INPUTS = (
    "theta",
    "phi",
    "u_knots",
    "t_wrap",
    "t_pull",
    "t_add",
    "t_spritz",
    "dt_spritz",
    "spritz_dose",
)


def _jvec(x) -> "juliacall.VectorValue":
    return juliacall.convert(
        jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    )


class InputSchema(BaseModel):
    """Inputs to the coupled cook.

    Differentiable: meat parameters ``theta``, pit parameters ``phi``, the
    damper schedule ``u_knots``, and the wrap / pull times. The rest are
    grids, tolerances, initial conditions (including the full-field meat
    state and pit state for receding-horizon handoff), rest-environment
    constants, and the pit humidity constants.
    """

    theta: Differentiable[Array[(N_THETA,), Float64]] = Field(
        description="Meat parameters [D_eff, k_m, h_c, L, k_mult, k_e, a_myosin]"
    )
    phi: Differentiable[Array[(N_PHI,), Float64]] = Field(
        description="Pit parameters [C_th, UA, r_max, dH_eff, u_leak, k_spread, a0]; "
        "a0 is the initial ignited fraction (state handoff enters here)"
    )
    u_knots: Differentiable[Array[(None,), Float64]] = Field(
        description="Damper schedule in [0, 1] on the uniform grid `u_tknots`"
    )
    u_tknots: Array[(None,), Float64] = Field(description="Damper knot times [s]")
    tsave: Array[(None,), Float64] = Field(description="Output save times [s]")
    t_wrap: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9, description="Texas-crutch wrap time [s]; 1e9 = never"
    )
    t_pull: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9, description="Pull time [s]; 1e9 = never (the meat then rests)"
    )
    t_add: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9,
        description="Mid-cook fuel addition time [s] (one chimney); 1e9 = none",
    )
    t_spritz: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9, description="First spritz [s]; 1e9 = no spritzing"
    )
    dt_spritz: Differentiable[Array[(), Float64]] = Field(
        default=3600.0, description="Spritz interval [s]"
    )
    spritz_dose: Differentiable[Array[(), Float64]] = Field(
        default=0.05, description="Water per spritz per exposed area [kg/m²]"
    )
    n_meat: float = Field(default=1.0, description="Identical briskets sharing the pit")
    abstol: float = Field(default=1e-6, description="Solver absolute tolerance")
    reltol: float = Field(default=1e-6, description="Solver relative tolerance")
    tpit0: float = Field(default=293.15, description="Initial pit temperature [K]")
    t_amb: float = Field(default=293.15, description="Ambient temperature [K]")
    m0: float = Field(default=7.0, description="Initial charcoal load [kg]")
    t0_meat: float = Field(default=281.15, description="Initial meat temperature [K]")
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
        default_factory=list, description="Initial fat field (len nx) or empty"
    )
    bark_b0: Float64 = Field(
        default=float("nan"), description="Initial bark B (NaN = 0)"
    )
    bark_s0: Float64 = Field(
        default=float("nan"), description="Initial bark S (NaN = 0)"
    )
    ring0: Float64 = Field(
        default=float("nan"),
        description="Initial smoke-ring depth, material coordinate [m] (NaN = 0)",
    )
    rest_t: float = Field(
        default=293.15, description="Rest environment temperature [K]"
    )
    rest_h: float = Field(
        default=1.5, description="Convective coefficient at rest [W/m²K]"
    )
    rest_eps: float = Field(default=0.10, description="Emissivity at rest")
    rest_eta: float = Field(default=0.9, description="Evaporation suppression at rest")
    n_shrink: float = Field(
        default=1.0 / 3.0, description="Thickness share of shrinkage"
    )
    afr: float = Field(default=20.0, description="Draft air-to-fuel ratio [kg/kg]")
    y_w: float = Field(default=0.28, description="Combustion water yield [kg/kg]")
    pan_kg_s: float = Field(default=0.0, description="Water-pan evaporation [kg/s]")
    omega_amb: float = Field(
        default=0.007, description="Ambient absolute humidity [kg/kg]"
    )


class OutputSchema(BaseModel):
    """Outputs of the coupled cook at ``tsave`` plus terminal quality and state."""

    tpit: Differentiable[Array[(None,), Float64]] = Field(
        description="Pit temperature [K]"
    )
    tcore: Differentiable[Array[(None,), Float64]] = Field(
        description="Core temperature [K]"
    )
    tsurf: Differentiable[Array[(None,), Float64]] = Field(
        description="Surface temperature [K]"
    )
    m_fuel: Differentiable[Array[(None,), Float64]] = Field(
        description="Fuel mass [kg]"
    )
    j_evap: Differentiable[Array[(None,), Float64]] = Field(
        description="Evaporative flux [kg/m²/s]"
    )
    q_meat: Differentiable[Array[(None,), Float64]] = Field(
        description="Heat drawn from the pit [W]"
    )
    c_field_final: Differentiable[Array[(NX,), Float64]] = Field(
        description="Final collagen field"
    )
    w_retained: Differentiable[Array[(), Float64]] = Field(
        description="Retained water / W0"
    )
    bark_b: Differentiable[Array[(None,), Float64]] = Field(
        description="Bark Maillard state"
    )
    bark_s: Differentiable[Array[(None,), Float64]] = Field(
        description="Bark smoke state"
    )
    l_star: Differentiable[Array[(None,), Float64]] = Field(description="Surface L*")
    juiciness: Differentiable[Array[(), Float64]] = Field(
        description="Perceptual juiciness"
    )
    fat_rendered: Differentiable[Array[(), Float64]] = Field(
        description="Rendered fat fraction"
    )
    l_thick: Differentiable[Array[(None,), Float64]] = Field(
        description="Half-thickness [m]"
    )
    ignited: Array[(None,), Float64] = Field(description="Ignited fuel fraction")
    mdot: Array[(None,), Float64] = Field(description="Burn rate [kg/s]")
    pv_pit: Array[(None,), Float64] = Field(
        description="Pit vapor pressure at the surface [Pa]"
    )
    pv_base: Array[(None,), Float64] = Field(description="Pit base vapor pressure [Pa]")
    evap_w: Array[(), Float64] = Field(
        description="Net evaporated water, condensation negative (dry-basis mean)"
    )
    drip_w: Array[(), Float64] = Field(description="Dripped water (dry-basis mean)")
    drip_f: Array[(), Float64] = Field(description="Drained fat (dry-basis mean)")
    t_field_final: Array[(NX,), Float64] = Field(description="Final T field (handoff)")
    w_field_final: Array[(NX,), Float64] = Field(description="Final W field (handoff)")
    f_field_final: Array[(NX,), Float64] = Field(
        description="Final fat field (handoff)"
    )
    ring: Array[(), Float64] = Field(
        description="Final smoke-ring depth, material coordinate [m] (handoff; "
        "× l_thick[-1]/L for the physical depth)"
    )
    spritz_w: Array[(), Float64] = Field(
        description="Spritz water absorbed by the meat (dry-basis mean); the rest flashed off"
    )


SOFT_FAIL = os.environ.get("BRISKET_SOFT_FAIL", "0") == "1"
_SENTINEL_T = 1.0e4
_SOFT_FAIL_COUNT = 0
_SOLVER_FAIL_MARKERS = (
    "solve failed",
    "retcode",
    "nan inputs",
    "dt_epsilon",
    "dt < eps",
    "domainerror",
)


def _is_solver_failure(err: BaseException) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _SOLVER_FAIL_MARKERS)


def _maybe_soft_fail(err: RuntimeError, what: str) -> None:
    """Re-raise programming errors; swallow classified solver failures when enabled."""
    global _SOFT_FAIL_COUNT
    if not SOFT_FAIL or not _is_solver_failure(err):
        raise err
    _SOFT_FAIL_COUNT += 1
    if _SOFT_FAIL_COUNT == 1 or _SOFT_FAIL_COUNT % 10 == 0:
        msg = str(err).encode("ascii", "replace").decode("ascii")
        print(f"SOFT_FAIL {what} #{_SOFT_FAIL_COUNT}: {msg}", flush=True)


def _own(x) -> np.ndarray:
    """Eager owned copy on the callback thread (XLA buffers are transient)."""
    return np.array(x, dtype=np.float64, copy=True)


def _packed(inputs: InputSchema):
    """Positional Julia arguments: (θ, φ, u, utk, tsave, t_wrap, t_pull, t_add, t_spritz, dt_spritz) + config tail."""
    theta, phi, u, utk, tsave = (
        _own(inputs.theta),
        _own(inputs.phi),
        _own(inputs.u_knots),
        _own(inputs.u_tknots),
        _own(inputs.tsave),
    )
    if u.shape != utk.shape:
        raise ValueError(f"DEGENERATE INPUT: u_knots {u.shape} vs u_tknots {utk.shape}")
    head = (
        theta,
        phi,
        u,
        utk,
        tsave,
        float(np.asarray(inputs.t_wrap)),
        float(np.asarray(inputs.t_pull)),
        float(np.asarray(inputs.t_add)),
        float(np.asarray(inputs.t_spritz)),
        float(np.asarray(inputs.dt_spritz)),
        float(np.asarray(inputs.spritz_dose)),
    )
    tail = (
        float(inputs.abstol),
        float(inputs.reltol),
        float(inputs.tpit0),
        float(inputs.t_amb),
        float(inputs.m0),
        float(inputs.t0_meat),
        _own(inputs.t_field0),
        _own(inputs.w_field0),
        _own(inputs.c_field0),
        _own(inputs.f_field0),
        float(inputs.bark_b0),
        float(inputs.bark_s0),
        float(inputs.ring0),
        float(inputs.rest_t),
        float(inputs.rest_h),
        float(inputs.rest_eps),
        float(inputs.rest_eta),
        float(inputs.n_shrink),
        float(inputs.afr),
        float(inputs.y_w),
        float(inputs.pan_kg_s),
        float(inputs.omega_amb),
        float(inputs.n_meat),
    )
    return head, tail


def _jl_args(head, tail):
    theta, phi, u, utk, tsave, t_wrap, t_pull, t_add, t_spritz, dt_spritz, m_sp = head
    jt = [_jvec(v) if isinstance(v, np.ndarray) else v for v in tail]
    return [
        _jvec(theta),
        _jvec(phi),
        _jvec(u),
        _jvec(utk),
        _jvec(tsave),
        t_wrap,
        t_pull,
        t_add,
        t_spritz,
        dt_spritz,
        m_sp,
    ], jt


def _stack_layout(nt: int) -> list[tuple[str, int]]:
    return (
        [(k, nt) for k in STACK_SERIES]
        + [(k, NX) for k in STACK_FIELDS]
        + [(k, 1) for k in STACK_SCALARS]
    )


def _input_slices(nu: int) -> dict[str, slice]:
    """Column ranges of each differentiable input in [θ; φ; u; t_wrap; t_pull; t_add; t_spritz; dt_spritz]."""
    b = N_THETA + N_PHI + nu
    return {
        "theta": slice(0, N_THETA),
        "phi": slice(N_THETA, N_THETA + N_PHI),
        "u_knots": slice(N_THETA + N_PHI, b),
        "t_wrap": slice(b, b + 1),
        "t_pull": slice(b + 1, b + 2),
        "t_add": slice(b + 2, b + 3),
        "t_spritz": slice(b + 3, b + 4),
        "dt_spritz": slice(b + 4, b + 5),
        "spritz_dose": slice(b + 5, b + 6),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """One coupled solve (Tesseract apply endpoint)."""
    head, tail = _packed(inputs)
    nt = len(head[4])

    def _run():
        jh, jt = _jl_args(head, tail)
        out = jl._mono_apply(*jh, *jt)
        # .copy(): no zero-copy views of Julia memory may cross threads
        return [np.asarray(v).copy() if np.ndim(v) else float(v) for v in out]

    try:
        vals = _on_julia_thread(_run)
    except RuntimeError as e:
        _maybe_soft_fail(e, "monolith apply")
        z, zx = np.zeros(nt), np.zeros(NX)
        vals = [
            np.full(nt, _SENTINEL_T),
            np.full(nt, _SENTINEL_T),
            np.full(nt, _SENTINEL_T),
            z,
            z,
            z,
            z,
            z,
            zx,
            0.0,
            z,
            z,
            np.full(nt, 52.0),
            0.0,
            0.0,
            z,
            z,
            z,
            0.0,
            0.0,
            0.0,
            np.full(NX, _SENTINEL_T),
            zx,
            zx,
            0.0,
            0.0,
        ]
    keys = (
        "tpit",
        "tcore",
        "tsurf",
        "m_fuel",
        "ignited",
        "mdot",
        "j_evap",
        "q_meat",
        "c_field_final",
        "w_retained",
        "bark_b",
        "bark_s",
        "l_star",
        "juiciness",
        "fat_rendered",
        "l_thick",
        "pv_pit",
        "pv_base",
        "evap_w",
        "drip_w",
        "drip_f",
        "t_field_final",
        "w_field_final",
        "f_field_final",
        "ring",
        "spritz_w",
    )
    return OutputSchema(**dict(zip(keys, vals, strict=True)))


def abstract_eval(abstract_inputs):
    """Output shapes/dtypes from ``tsave.shape[0]`` (abstract_eval endpoint)."""
    nt = abstract_inputs.tsave.shape[0]
    vec = lambda n: ShapeDType(shape=(n,), dtype="float64")
    sc = ShapeDType(shape=(), dtype="float64")
    out = {k: vec(nt) for k in STACK_SERIES}
    out.update({k: vec(NX) for k in STACK_FIELDS})
    out.update({k: sc for k in STACK_SCALARS})
    out.update(
        {"ignited": vec(nt), "mdot": vec(nt), "pv_pit": vec(nt), "pv_base": vec(nt)}
    )
    out.update({"evap_w": sc, "drip_w": sc, "drip_f": sc})
    out.update(
        {"t_field_final": vec(NX), "w_field_final": vec(NX), "f_field_final": vec(NX)}
    )
    out["ring"] = sc
    out["spritz_w"] = sc
    return out


def _selected_columns(inputs: InputSchema, names) -> tuple[list[int], dict[str, slice]]:
    nu = len(inputs.u_knots)
    sl = _input_slices(nu)
    cols: list[int] = []
    for n in DIFF_INPUTS:
        if n in names:
            cols.extend(range(sl[n].start, sl[n].stop))
    return cols, sl


def _stack_jacobian(inputs: InputSchema, cols: list[int]) -> np.ndarray:
    """Jacobian of the stacked outputs w.r.t. the selected 0-based columns."""
    head, tail = _packed(inputs)

    def _run():
        jh, jt = _jl_args(head, tail)
        sel = _jvec(np.asarray(cols, dtype=np.float64) + 1.0)
        return np.asarray(jl._mono_jac(*jh, sel, *jt)).copy()

    return _on_julia_thread(_run)


def _split_stack(vec: np.ndarray, nt: int) -> dict[str, np.ndarray]:
    out, i = {}, 0
    for k, n in _stack_layout(nt):
        out[k] = vec[i : i + n] if n > 1 or k in STACK_FIELDS else vec[i]
        i += n
    return out


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    """Jᵀ·cotangent over the requested inputs (one forward-mode Jacobian pass)."""
    nt = len(inputs.tsave)
    ct = []
    for k, n in _stack_layout(nt):
        v = cotangent_vector.get(k) if k in vjp_outputs else None
        ct.append(np.zeros(n) if v is None else np.atleast_1d(_own(v)).reshape(n))
    ct = np.concatenate(ct)
    cols, sl = _selected_columns(inputs, vjp_inputs)
    try:
        J = _stack_jacobian(inputs, cols)
    except RuntimeError as e:
        _maybe_soft_fail(e, "monolith vjp")
        J = np.zeros((len(ct), len(cols)))
    g = J.T @ ct
    if np.isnan(g).any():
        cap = Path(tempfile.gettempdir()) / "monolith_vjp_nan_capture.npz"
        np.savez(
            cap,
            theta=_own(inputs.theta),
            phi=_own(inputs.phi),
            u=_own(inputs.u_knots),
            ct=ct,
        )
        raise FloatingPointError(
            f"monolith VJP produced NaN gradients; call captured to {cap}"
        )
    out, i = {}, 0
    for n in DIFF_INPUTS:
        if n in vjp_inputs:
            width = sl[n].stop - sl[n].start
            block = g[i : i + width]
            out[n] = (
                block
                if n in ("theta", "phi", "u_knots")
                else np.asarray(block[0], dtype=np.float64)
            )
            i += width
    return out


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict,
):
    """J·tangent over the requested inputs (one Dual solve)."""
    nu = len(inputs.u_knots)
    sl = _input_slices(nu)
    dv = np.zeros(N_THETA + N_PHI + nu + 6)
    for n in DIFF_INPUTS:
        if n in jvp_inputs and n in tangent_vector:
            dv[sl[n]] = np.atleast_1d(_own(tangent_vector[n])).reshape(-1)
    head, tail = _packed(inputs)
    nt = len(head[4])

    def _run():
        jh, jt = _jl_args(head, tail)
        return np.asarray(jl._mono_jvp(*jh, _jvec(dv), *jt)).copy()

    try:
        d = _on_julia_thread(_run)
    except RuntimeError as e:
        _maybe_soft_fail(e, "monolith jvp")
        d = np.zeros(sum(n for _, n in _stack_layout(nt)))
    parts = _split_stack(d, nt)
    return {k: v for k, v in parts.items() if k in jvp_outputs}


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    """Dense Jacobian blocks {output: {input: array(out_shape + in_shape)}}."""
    nt = len(inputs.tsave)
    cols, sl = _selected_columns(inputs, jac_inputs)
    J = _stack_jacobian(inputs, cols)
    # column offsets of each selected input inside J
    offs, i = {}, 0
    for n in DIFF_INPUTS:
        if n in jac_inputs:
            w = sl[n].stop - sl[n].start
            offs[n] = (i, i + w)
            i += w
    out = {}
    r = 0
    for k, n in _stack_layout(nt):
        rows = J[r : r + n, :]
        r += n
        if k not in jac_outputs:
            continue
        blocks = {}
        for name, (a, b) in offs.items():
            blk = rows[:, a:b]
            if k in STACK_SCALARS:
                blk = blk[0]
            if name in (
                "t_wrap",
                "t_pull",
                "t_add",
                "t_spritz",
                "dt_spritz",
                "spritz_dose",
            ):
                blk = blk[..., 0]
            blocks[name] = np.array(blk, dtype=np.float64)
        out[k] = blocks
    return out
