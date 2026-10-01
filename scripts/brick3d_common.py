"""The 3-D shadow's shared half: the plan and the calibrated MAP loaded as the 2-D shadow loads them, and one
3-D cook of the plan (scripts/shadow3d.jl) run on the slab2d Tesseract's Julia session (the repository's Julia
and project, one process-wide Julia thread), every array copied to numpy on that thread.

Used by scripts/run_shadow3d.py (the report) and scripts/plot_brick3d.py (the figures).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

from brisket_app.calibrate import PHI_SITES, PHI_SITES_IGNITION, THETA_SITES  # noqa: E402
from brisket_app.coupled import HR  # noqa: E402

_API = None


def julia_session():
    """The slab2d Tesseract's Julia session, with scripts/shadow3d.jl included (once)."""
    global _API
    if _API is None:
        path = REPO / "tesseracts" / "slab2d" / "tesseract_api.py"
        spec = importlib.util.spec_from_file_location("slab2d_tesseract_api", path)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        src = (REPO / "scripts" / "shadow3d.jl").as_posix()
        api._on_julia_thread(lambda: api.jl.seval(f'include(raw"{src}")'))
        _API = api
    return _API


def load_case(plan_name: str):
    """The calibrated MAP (θ, φ) and the plan's schedule and decisions, in seconds."""
    mp = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([mp[n] for n, _ in THETA_SITES], dtype=float)
    phi = np.array([mp[n] for n, _ in PHI_SITES] + [mp.get(n, d.mean) for n, d in PHI_SITES_IGNITION],
                   dtype=float)
    plan = json.loads((REPO / "data" / "control" / plan_name).read_text())
    u = np.asarray(plan.get("u_robust", plan.get("u_opt")), dtype=float)
    t_end = float(plan.get("cook_h", 11.0)) * HR

    def hours(key):
        v = plan.get(f"{key}_robust_h", plan.get(f"{key}_plan_h"))
        return float(v) * HR if v is not None else 1.0e9

    dts = plan.get("dt_spritz_robust_min", plan.get("dt_spritz_plan_min"))
    decisions = {
        "t_wrap": hours("t_wrap"), "t_pull": hours("t_pull"), "t_add": hours("t_add"),
        "t_spritz": hours("t_spritz"), "dt_spritz": float(dts) * 60.0 if dts is not None else 3600.0,
    }
    rest = plan.get("rest") or {}
    return theta, phi, u, t_end, decisions, rest


SERIES = ("Tcore", "Tcold", "Tsurf", "bark_b", "bark_s", "bark_std", "ring", "l_star", "l_thick", "mass", "area_factor")
FIELDS = ("T_field", "W_field", "C_field", "F_field", "salt_field", "bark_dist", "smoke_dist", "ring_dist")
SCALARS = ("w_retained", "juiciness", "fat_rendered", "bitterness")


def simulate(plan_name: str, nx: int, ny: int, nz: int, t_salt_s: float = 0.0, dt_save: float = 600.0,
             aspect: float = 2.0):
    """One 3-D cook of the plan to serve: the piece's series, fields and scalars, the pit series, the mesh
    (coordinates, dims, extents, the exposed nodes with their areas) and the geometry, as numpy."""
    theta, phi, u, t_end, dec, rest = load_case(plan_name)
    tsave = np.arange(0.0, t_end + 1.0, dt_save)
    utk = np.linspace(0.0, t_end, len(u))
    api = julia_session()

    def _run():
        jl = api.jl
        s = jl.shadow3d(
            api._jvec(theta), api._jvec(phi), api._jvec(u), api._jvec(utk), api._jvec(tsave),
            nx=nx, ny=ny, nz=nz, aspect=aspect, t_salt=t_salt_s, **dec,
            rest_t=float(rest.get("rest_t", 293.15)), rest_h=float(rest.get("rest_h", 1.5)),
            rest_eps=float(rest.get("rest_eps", 0.10)), rest_eta=float(rest.get("rest_eta", 0.9)),
        )
        p = jl.getproperty(s, jl.Symbol("piece"))
        get = lambda obj, name: np.array(jl.getproperty(obj, jl.Symbol(name)), copy=True)
        out = {n: get(p, n) for n in SERIES + FIELDS}
        out.update({n: float(get(p, n)) for n in SCALARS})
        for n in ("Tpit", "m_fuel"):
            out[n] = get(s, n)
        mesh = jl.getproperty(s, jl.Symbol("mesh"))
        out["coords"] = get(mesh, "coords")
        out["dims"] = tuple(int(d) for d in jl.getproperty(mesh, jl.Symbol("dims")))
        out["extents"] = tuple(float(e) for e in jl.getproperty(mesh, jl.Symbol("extents")))
        bnd = jl.getproperty(mesh, jl.Symbol("boundary"))
        out["boundary_nodes"] = np.array([int(b[0]) for b in bnd]) - 1        # 0-based
        out["boundary_areas"] = np.array([float(b[1]) for b in bnd])
        out["core"] = int(jl.getproperty(mesh, jl.Symbol("core"))) - 1
        geo = jl.getproperty(s, jl.Symbol("geometry"))
        out["geometry"] = {k: float(jl.getproperty(geo, jl.Symbol(k))) for k in ("thickness", "width", "length")}
        return out

    sim = api._on_julia_thread(_run)
    sim.update(t=tsave, plan=plan_name, decisions=dec, grid=f"{nx}x{ny}x{nz}")
    return sim
