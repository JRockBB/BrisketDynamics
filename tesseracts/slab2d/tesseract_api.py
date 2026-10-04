"""Tesseract API: the 2-D axisymmetric cook (pit + r-z meat, one solve).

The fidelity rung above the 1-D monolith: `coupled_solve_2d` co-integrates
the six pit states with the 2-D ADI meat (shrink metrics, dry-brine salt
field, spritz, wrap, PULL and rest, grain anisotropy, point-over-flat,
smoke deposition and the smoke ring on every face) and carries the pit
events (a fuel addition or split train, the lid lifts), and the stacked
forward-mode endpoints differentiate w.r.t. theta2, phi, the damper
schedule and the seven decision scalars (wrap, salt, spritz start and
interval, pull, fuel addition and split interval). The solve runs to SERVE: after `t_pull` the
meat rests in the configured environment while the pit runs on, exactly
as the 1-D plan is scored. Intended use: full-fidelity SHADOW SCORING of
1-D plans (true cold-spot safety, bark distribution) — the planners
themselves stay on the 1-D monolith. Geometry has no default: pass the
radius and height (see `brisket_app.shadow2d.equivalent_radius`).
"""

import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import filelock
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

REPO_ROOT = Path(__file__).resolve().parents[2]

if sys.platform == "win32" and not os.environ.get("PYTHON_JULIAPKG_EXE"):
    # juliacall's own Julia must be the one the repository's environments were instantiated with (`julia` on PATH
    # through juliaup); see tesseracts/monolith/tesseract_api.py and docs/install.md (Windows)
    import shutil as _shutil
    import subprocess as _subprocess

    _julia = _shutil.which("julia")
    if _julia:
        try:
            _exe = _subprocess.run(
                [_julia, "-e", 'print(joinpath(Sys.BINDIR, "julia.exe"))'],
                capture_output=True, text=True, timeout=120, check=True,
            ).stdout.strip()
        except Exception as _e:  # noqa: BLE001 - juliapkg then chooses; say so
            print(f"tesseract_api: could not resolve `julia` for juliacall ({_e}); juliapkg will choose its own", file=sys.stderr)
            _exe = ""
        if _exe:
            os.environ["PYTHON_JULIAPKG_EXE"] = _exe
    # and its project: the repository's own (PythonCall is a direct dependency of the package), so that one
    # manifest decides every version; a project of juliacall's own loads Preferences at another version and the
    # package's dependencies then cannot precompile inside the same session
    os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(REPO_ROOT))

_DIRECT = os.environ.get("BRISKET_JL_DIRECT", "0") == "1"
# the ONE process-wide Julia thread, shared with the monolith API:
# libjulia initializes on the first module's executor thread, and a
# second module driving it from its own thread deadlocks the process
_EXEC_ATTR = "_brisket_julia_executor"
if getattr(sys, _EXEC_ATTR, None) is None:
    setattr(
        sys,
        _EXEC_ATTR,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="juliacall"),
    )
_JL_THREAD: ThreadPoolExecutor = getattr(sys, _EXEC_ATTR)
_INIT_LOCK = threading.Lock()
juliacall = None
jl = None


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

    jl = juliacall.newmodule("Slab2dTess")
    _project = os.environ.get("JULIA_PROJECT") or str(REPO_ROOT)
    jl.seval(f'import Pkg; Pkg.activate(raw"{_project}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jl.seval(
        """
        function _s2_kw(radius, height, nr, nz, h_step, n_meat, k_grain,
                        d_grain, f_top, z_split, t0_meat, tpit0, t_amb, m0,
                        afr, y_w, pan, omega, rest_t, rest_h, rest_eps,
                        rest_eta, t_load, T_load)
            cfg = MeatConfig(T0=t0_meat, T_rest=rest_t, h_rest=rest_h,
                             ε_rest=rest_eps, η_rest=rest_eta)
            pit = PitConfig(T_amb=t_amb, Tpit0=tpit0, m0=m0)
            hum = BrisketDynamics.Humidity(afr=afr, y_w=y_w, pan_kg_s=pan,
                                           omega_amb=omega)
            return (; R=radius, Hz=height, nr=Int(nr), nz=Int(nz), h=h_step,
                    cfg, pit, hum, n_meat, k_grain, D_grain=d_grain,
                    F_top=f_top, z_split, t_load=float(t_load), T_load=float(T_load))
        end
        # the load (data on the 2-D path): T_load > 0 [K] puts the meat on when this pit, preheating empty under the
        # schedule, first reaches it; else t_load [s from lighting; <= -1e8 in from the start]
        function _s2_resolve(kw, φ, u, utk)
            t_ld = kw.T_load > 0 ?
                first(preheat_time(kw.T_load, collect(Float64, φ), collect(Float64, u), collect(Float64, utk);
                                   cfg=kw.cfg, pit=kw.pit, hum=kw.hum, Tpit0=kw.pit.Tpit0, mfuel0=kw.pit.m0)) :
                kw.t_load
            return merge(Base.structdiff(kw, NamedTuple{(:T_load,)}), (; t_load=t_ld))
        end
        # dt_add: 0 = one chimney (Inf), as in the monolith Tesseract
        _s2_dta(dt_add) = dt_add > 0 ? float(dt_add) : Inf
        function _s2_apply(θ2, φ, u, utk, tsave, t_wrap, t_salt, t_spritz,
                           dt_spritz, t_pull, t_add, dt_add, kwargs...)
            kw = _s2_resolve(_s2_kw(kwargs...), φ, u, utk)
            o = coupled_solve_2d(θ2, φ, u, utk, tsave; t_wrap, t_salt,
                                 t_spritz, dt_spritz, t_pull, t_add,
                                 dt_add=_s2_dta(dt_add), kw...)
            return (o.Tpit, o.Tcore, o.Tcold, o.m_fuel, o.w_retained, o.C_min,
                    o.bark_mean, o.bark_std, o.salt_mean, o.R_eff, o.Hz_eff,
                    o.smoke_mean, o.ring_mean, o.l_star_mean, o.bitterness,
                    o.fat_rendered, o.juiciness, o.pv_pit, o.j_evap, kw.t_load)
        end
        function _s2_jac(θ2, φ, u, utk, tsave, t_wrap, t_salt, t_spritz,
                         dt_spritz, t_pull, t_add, dt_add, sel, kwargs...)
            kw = _s2_resolve(_s2_kw(kwargs...), φ, u, utk)
            return slab2d_stack_jacobian(θ2, φ, u, utk, tsave, t_wrap, t_salt,
                                         t_spritz, dt_spritz, t_pull,
                                         collect(Int, sel); t_add,
                                         dt_add=_s2_dta(dt_add), kw...)
        end
        function _s2_jvp(θ2, φ, u, utk, tsave, t_wrap, t_salt, t_spritz,
                         dt_spritz, t_pull, t_add, dt_add, dv, kwargs...)
            kw = _s2_resolve(_s2_kw(kwargs...), φ, u, utk)
            return slab2d_stack_jvp(θ2, φ, u, utk, tsave, t_wrap, t_salt,
                                    t_spritz, dt_spritz, t_pull,
                                    collect(Float64, dv); t_add,
                                    dt_add=_s2_dta(dt_add), kw...)
        end
        """
    )


with filelock.FileLock(
    Path(tempfile.gettempdir()) / "brisket_julia_init.lock", timeout=600
):
    _on_julia_thread(_init)

N_THETA2, N_PHI = 7, 7
STACK_SERIES = ("tpit", "tcore", "tcold", "m_fuel")
STACK_SCALARS = (
    "w_retained",
    "c_min",
    "bark_mean",
    "bark_std",
    "salt_mean",
    "r_eff",
    "hz_eff",
    "smoke_mean",
    "ring_mean",
    "l_star_mean",
    "bitterness",
    "fat_rendered",
    "juiciness",
)
DIFF_INPUTS = (
    "theta2",
    "phi",
    "u_knots",
    "t_wrap",
    "t_salt",
    "t_spritz",
    "dt_spritz",
    "t_pull",
    "t_add",
    "dt_add",
)
# the decision scalars (their Jacobian blocks are scalars, not vectors)
SCALAR_INPUTS = DIFF_INPUTS[3:]


def _jvec(x):
    return juliacall.convert(
        jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    )


class InputSchema(BaseModel):
    """Inputs to the 2-D coupled cook (shadow-scoring fidelity rung)."""

    theta2: Differentiable[Array[(N_THETA2,), Float64]] = Field(
        description="Meat parameters [D_eff, k_m, h_c, k_mult, k_e, a_myosin, m_conv] "
        "(theta without L; geometry is radius/height)"
    )
    phi: Differentiable[Array[(N_PHI,), Float64]] = Field(
        description="Pit parameters [C_th, UA, r_max, dH_eff, u_leak, k_spread, a0]"
    )
    u_knots: Differentiable[Array[(None,), Float64]] = Field(
        description="Damper schedule in [0, 1] on `u_tknots`"
    )
    u_tknots: Array[(None,), Float64] = Field(description="Damper knot times [s]")
    tsave: Array[(None,), Float64] = Field(description="Output save times [s]")
    t_wrap: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9, description="Wrap time [s]; 1e9 = never"
    )
    t_salt: Differentiable[Array[(), Float64]] = Field(
        default=0.0, description="Dry-brine time before the cook [s]; 0 = unsalted"
    )
    t_spritz: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9, description="First spritz [s]; 1e9 = none"
    )
    dt_spritz: Differentiable[Array[(), Float64]] = Field(
        default=3600.0, description="Spritz interval [s]"
    )
    t_pull: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9,
        description="Pull time [s]; the meat rests to the horizon after it; 1e9 = never",
    )
    t_add: Differentiable[Array[(), Float64]] = Field(
        default=1.0e9,
        description="Mid-cook fuel addition time [s] (one chimney, and its lid "
        "lift); 1e9 = none",
    )
    dt_add: Differentiable[Array[(), Float64]] = Field(
        default=0.0,
        description="Wood-split interval [s]: a finite value turns t_add into a "
        "pulse train (one split of m_chimney every dt_add); 0 = single addition",
    )
    radius: float = Field(
        description="Cylinder radius [m]; no default — derive it from the cut "
        "(shadow2d.equivalent_radius: the 1-D exposed area A_meat)"
    )
    height: float = Field(description="Cylinder full height [m] (2L for a flat)")
    nr: int = Field(default=16, description="Radial nodes")
    nz: int = Field(default=14, description="Axial nodes")
    h_step: float = Field(default=30.0, description="Fixed step [s]")
    n_meat: float = Field(default=1.0, description="Identical cuts sharing the pit")
    k_grain: float = Field(default=1.0, description="In-plane conduction multiplier")
    d_grain: float = Field(default=1.0, description="In-plane moisture multiplier")
    f_top: float = Field(
        default=float("nan"),
        description="Point-muscle fat fraction above z_split (NaN = uniform cut)",
    )
    z_split: float = Field(default=1.0, description="Point/flat split height fraction")
    t0_meat: float = Field(default=281.15, description="Initial meat temperature [K]")
    tpit0: float = Field(default=293.15, description="Initial pit temperature [K]")
    t_amb: float = Field(default=293.15, description="Ambient temperature [K]")
    m0: float = Field(default=7.0, description="Initial charcoal load [kg]")
    afr: float = Field(default=20.0, description="Air-to-fuel ratio of the draft")
    y_w: float = Field(default=0.28, description="Combustion water yield [kg/kg]")
    pan_kg_s: float = Field(default=0.0, description="Water-pan evaporation [kg/s]")
    omega_amb: float = Field(default=0.007, description="Ambient humidity [kg/kg]")
    rest_t: float = Field(default=293.15, description="Rest environment temperature [K]")
    rest_h: float = Field(default=1.5, description="Convective coefficient at rest [W/m²K]")
    rest_eps: float = Field(default=0.10, description="Emissivity at rest")
    rest_eta: float = Field(default=0.9, description="Evaporation suppression at rest")
    t_load: float = Field(
        default=-1.0e9,
        description="When the meat goes on [s after lighting]: before it the pit preheats empty and the meat holds "
        "its raw state; <= -1e8 = in from the start. Data, not differentiable; ignored when T_load > 0",
    )
    T_load: float = Field(
        default=0.0,
        description="The pit temperature the meat goes in at [K]: when > 0 the load is when this pit, preheating "
        "empty under the schedule, first reaches it; 0 = use t_load",
    )


class OutputSchema(BaseModel):
    """The 2-D cook's series and quality scalars."""

    tpit: Differentiable[Array[(None,), Float64]] = Field(description="Pit T [K]")
    tcore: Differentiable[Array[(None,), Float64]] = Field(
        description="Axis mid-height T [K]"
    )
    tcold: Differentiable[Array[(None,), Float64]] = Field(
        description="TRUE cold spot: grid-minimum T [K]"
    )
    m_fuel: Differentiable[Array[(None,), Float64]] = Field(
        description="Fuel mass [kg]"
    )
    w_retained: Differentiable[Array[(), Float64]] = Field(
        description="Volume-weighted retained-water fraction"
    )
    c_min: Differentiable[Array[(), Float64]] = Field(
        description="Minimum collagen conversion at serve (the last-done spot)"
    )
    bark_mean: Differentiable[Array[(), Float64]] = Field(
        description="Area-weighted Maillard bark mean over the exposed faces"
    )
    bark_std: Differentiable[Array[(), Float64]] = Field(
        description="Bark spread (burnt ends live in the tail)"
    )
    salt_mean: Differentiable[Array[(), Float64]] = Field(
        description="Volume-weighted mean salt, dry basis"
    )
    r_eff: Differentiable[Array[(), Float64]] = Field(
        description="Current (shrunk) radius [m]"
    )
    hz_eff: Differentiable[Array[(), Float64]] = Field(
        description="Current (shrunk) height [m]"
    )
    smoke_mean: Differentiable[Array[(), Float64]] = Field(
        description="Area-weighted smoke deposition over the exposed faces"
    )
    ring_mean: Differentiable[Array[(), Float64]] = Field(
        description="Area-weighted smoke-ring depth [m], material coordinate"
    )
    l_star_mean: Differentiable[Array[(), Float64]] = Field(
        description="Surface lightness L* at the face-mean bark and smoke"
    )
    bitterness: Differentiable[Array[(), Float64]] = Field(
        description="Area-mean dirty-smoke deposition integral"
    )
    fat_rendered: Differentiable[Array[(), Float64]] = Field(
        description="Rendered share of the cut's initial fat (per-node rendering)"
    )
    juiciness: Differentiable[Array[(), Float64]] = Field(
        description="Retained water plus retained rendered fat (the 1-D score)"
    )
    pv_pit: Array[(None,), Float64] = Field(description="Pit vapor pressure [Pa]")
    j_evap: Array[(None,), Float64] = Field(description="Meat evaporation [kg/s]")
    t_load_used: Array[(), Float64] = Field(
        description="When the meat went on [s after lighting]: t_load, or the preheat's time to T_load"
    )


def _packed(inputs: InputSchema):
    head = (
        np.asarray(inputs.theta2, dtype=np.float64),
        np.asarray(inputs.phi, dtype=np.float64),
        np.asarray(inputs.u_knots, dtype=np.float64),
        np.asarray(inputs.u_tknots, dtype=np.float64),
        np.asarray(inputs.tsave, dtype=np.float64),
        float(np.asarray(inputs.t_wrap)),
        float(np.asarray(inputs.t_salt)),
        float(np.asarray(inputs.t_spritz)),
        float(np.asarray(inputs.dt_spritz)),
        float(np.asarray(inputs.t_pull)),
        float(np.asarray(inputs.t_add)),
        float(np.asarray(inputs.dt_add)),
    )
    tail = (
        float(inputs.radius),
        float(inputs.height),
        int(inputs.nr),
        int(inputs.nz),
        float(inputs.h_step),
        float(inputs.n_meat),
        float(inputs.k_grain),
        float(inputs.d_grain),
        float(inputs.f_top),
        float(inputs.z_split),
        float(inputs.t0_meat),
        float(inputs.tpit0),
        float(inputs.t_amb),
        float(inputs.m0),
        float(inputs.afr),
        float(inputs.y_w),
        float(inputs.pan_kg_s),
        float(inputs.omega_amb),
        float(inputs.rest_t),
        float(inputs.rest_h),
        float(inputs.rest_eps),
        float(inputs.rest_eta),
        float(inputs.t_load),
        float(inputs.T_load),
    )
    return head, tail


def _jl_args(head):
    t2, phi, u, utk, tsave, *scalars = head
    return [_jvec(t2), _jvec(phi), _jvec(u), _jvec(utk), _jvec(tsave),
            *scalars]


def _stack_layout(nt: int):
    return [(k, nt) for k in STACK_SERIES] + [(k, 1) for k in STACK_SCALARS]


def _input_slices(nu: int) -> dict[str, slice]:
    b = N_THETA2 + N_PHI + nu
    return {
        "theta2": slice(0, N_THETA2),
        "phi": slice(N_THETA2, N_THETA2 + N_PHI),
        "u_knots": slice(N_THETA2 + N_PHI, b),
        "t_wrap": slice(b, b + 1),
        "t_salt": slice(b + 1, b + 2),
        "t_spritz": slice(b + 2, b + 3),
        "dt_spritz": slice(b + 3, b + 4),
        "t_pull": slice(b + 4, b + 5),
        "t_add": slice(b + 5, b + 6),
        "dt_add": slice(b + 6, b + 7),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """One 2-D coupled solve (apply endpoint)."""
    head, tail = _packed(inputs)

    def _run():
        out = jl._s2_apply(*_jl_args(head), *tail)
        return [np.asarray(v).copy() if np.ndim(v) else float(v) for v in out]

    vals = _on_julia_thread(_run)
    keys = (
        "tpit", "tcore", "tcold", "m_fuel", "w_retained", "c_min",
        "bark_mean", "bark_std", "salt_mean", "r_eff", "hz_eff",
        "smoke_mean", "ring_mean", "l_star_mean", "bitterness",
        "fat_rendered", "juiciness", "pv_pit", "j_evap", "t_load_used",
    )
    return OutputSchema(**dict(zip(keys, vals, strict=True)))


def abstract_eval(abstract_inputs):
    """Output shapes/dtypes from ``tsave.shape[0]``."""
    nt = abstract_inputs.tsave.shape[0]
    vec = lambda n: ShapeDType(shape=(n,), dtype="float64")
    sc = ShapeDType(shape=(), dtype="float64")
    out = {k: vec(nt) for k in STACK_SERIES}
    out.update({k: sc for k in STACK_SCALARS})
    out.update({"pv_pit": vec(nt), "j_evap": vec(nt), "t_load_used": sc})
    return out


def _selected_columns(inputs: InputSchema, names):
    nu = len(inputs.u_knots)
    sl = _input_slices(nu)
    cols: list[int] = []
    for n in DIFF_INPUTS:
        if n in names:
            cols.extend(range(sl[n].start, sl[n].stop))
    return cols, sl


def _stack_jacobian(inputs: InputSchema, cols):
    head, tail = _packed(inputs)

    def _run():
        sel = _jvec(np.asarray(cols, dtype=np.float64) + 1.0)
        return np.asarray(jl._s2_jac(*_jl_args(head), sel, *tail)).copy()

    return _on_julia_thread(_run)


def _split_stack(vec: np.ndarray, nt: int):
    out, i = {}, 0
    for k, n in _stack_layout(nt):
        out[k] = vec[i : i + n] if k in STACK_SERIES else vec[i]
        i += n
    return out


def jacobian(inputs: InputSchema, jac_inputs: set, jac_outputs: set):
    """Dense Jacobian blocks {output: {input: array}}."""
    nt = len(inputs.tsave)
    cols, sl = _selected_columns(inputs, jac_inputs)
    J = _stack_jacobian(inputs, cols)
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
            if name in SCALAR_INPUTS:
                blk = blk[..., 0]
            blocks[name] = np.array(blk, dtype=np.float64)
        out[k] = blocks
    return out


def jacobian_vector_product(
    inputs: InputSchema, jvp_inputs: set, jvp_outputs: set, tangent_vector: dict
):
    """J·tangent (one Dual solve)."""
    nu = len(inputs.u_knots)
    sl = _input_slices(nu)
    dv = np.zeros(N_THETA2 + N_PHI + nu + len(SCALAR_INPUTS))
    for n in DIFF_INPUTS:
        if n in jvp_inputs and n in tangent_vector:
            dv[sl[n]] = np.atleast_1d(
                np.asarray(tangent_vector[n], dtype=np.float64)
            ).reshape(-1)
    head, tail = _packed(inputs)
    nt = len(head[4])

    def _run():
        return np.asarray(jl._s2_jvp(*_jl_args(head), _jvec(dv), *tail)).copy()

    d = _on_julia_thread(_run)
    parts = _split_stack(d, nt)
    return {k: v for k, v in parts.items() if k in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema, vjp_inputs: set, vjp_outputs: set, cotangent_vector: dict
):
    """Jᵀ·cotangent (one forward Jacobian pass)."""
    nt = len(inputs.tsave)
    ct = []
    for k, n in _stack_layout(nt):
        v = cotangent_vector.get(k) if k in vjp_outputs else None
        ct.append(
            np.zeros(n) if v is None
            else np.atleast_1d(np.asarray(v, dtype=np.float64)).reshape(n)
        )
    ct = np.concatenate(ct)
    cols, sl = _selected_columns(inputs, vjp_inputs)
    J = _stack_jacobian(inputs, cols)
    g = J.T @ ct
    out, i = {}, 0
    for n in DIFF_INPUTS:
        if n in vjp_inputs:
            w = sl[n].stop - sl[n].start
            block = g[i : i + w]
            out[n] = (
                block
                if n in ("theta2", "phi", "u_knots")
                else np.asarray(block[0], dtype=np.float64)
            )
            i += w
    return out
