"""The 3-D shadow's shared half: the plan and the calibrated MAP loaded as the 2-D shadow loads them, one 3-D cook of
the plan on the trimmed brisket flat (scripts/shadow3d.jl) run on the slab2d Tesseract's Julia session (the
repository's Julia and project, one process-wide Julia thread, every array copied to numpy on that thread), and the
flat's fields on its node grid.

The flat's mesh (src/meats/grid3d.jl `_flat_mesh`): nodes on a mapped grid, `nl` along the length (from the thick
end), `nw` across the half width (from the centre plane), `nt` up the thickness (from the grate); the node order is
length fastest. The exposed entries come in this order: the top and the grate side alternating per node of the
footprint, the outer side, the thick and the thin end alternating per node of the end face.

Used by scripts/run_shadow3d.py (the report), plot_shadow3d.py (the figures) and render_shadow3d.py (the 3-D views).
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
    T_load = plan.get("T_load_robust", plan.get("T_load_plan"))
    decisions = {
        "t_wrap": hours("t_wrap"), "t_pull": hours("t_pull"), "t_add": hours("t_add"),
        "t_spritz": hours("t_spritz"), "dt_spritz": float(dts) * 60.0 if dts is not None else 3600.0,
        # the meat goes on when the shadow's own pit reaches the plan's load temperature (from lighting)
        "T_load": float(T_load) if T_load else 0.0,
    }
    rest = plan.get("rest") or {}
    return theta, phi, u, t_end, decisions, rest


SERIES = ("Tcore", "Tcold", "Tsurf", "bark_b", "bark_s", "bark_std", "ring", "l_star", "l_thick", "mass", "area_factor")
FIELDS = ("T_field", "W_field", "C_field", "F_field", "salt_field", "bark_dist", "smoke_dist", "ring_dist")
SCALARS = ("w_retained", "juiciness", "fat_rendered", "bitterness")
DIMS = ("length", "width_thick", "width_thin", "thickness_thick", "thickness_thin", "cap")


def load_shape(path):
    """A reconstructed cut (tools/cutvision's shape.json): its labels (0 empty, 1 lean, 2 fat, 3 bone) as an
    (nx, ny, nz) array in the reconstruction's column-major order, its cell size [m], meat and metadata. A bare name
    reads data/cut/<name>/shape.json."""
    p = Path(path)
    if not p.exists():
        p = REPO / "data" / "cut" / str(path) / "shape.json"
    meta = json.loads(p.read_text())
    runs = np.asarray(meta["labels_rle"], dtype=np.int64).reshape(-1, 2)
    flat = np.repeat(runs[:, 0].astype(np.int8), runs[:, 1])
    dims = tuple(int(d) for d in meta["dims"])
    if flat.size != int(np.prod(dims)):
        raise ValueError(f"{p}: {flat.size} labels for a {dims} grid")
    return {"labels": flat.reshape(dims, order="F"), "h": float(meta["voxel_mm"]) / 1000.0, "meat": meta["meat"],
            "name": p.parent.name, "path": str(p), "meta": meta}


def simulate(plan_name: str, nl: int = 12, nw: int = 6, nt: int = 8, t_salt_s: float = 0.0, dt_save: float = 600.0,
             shape=None, coarsen: int = 4):
    """One 3-D cook of the plan to serve, on the flat or (`shape`, a `load_shape` dict) on a reconstructed cut with
    cells `coarsen` times the reconstruction's: the piece's series, fields and scalars, the pit series, the mesh
    (coordinates, dims, the exposed nodes with their areas, each node's material and bone weight, the faces'
    directions for a reconstructed cut) and the flat's dimensions, as numpy."""
    theta, phi, u, t_end, dec, rest = load_case(plan_name)
    tsave = np.arange(0.0, t_end + 1.0, dt_save)
    utk = np.linspace(0.0, t_end, len(u))
    api = julia_session()

    def _run():
        jl = api.jl
        extra = {}
        if shape is not None:
            lab = shape["labels"]
            extra = dict(shape_labels=api._jvec(lab.reshape(-1, order="F").astype(np.float64)),
                         shape_dims=api._jvec(np.asarray(lab.shape, dtype=np.float64)), shape_h=shape["h"],
                         meat=jl.Symbol(shape["meat"]), coarsen=int(coarsen))
        s = jl.shadow3d(
            api._jvec(theta), api._jvec(phi), api._jvec(u), api._jvec(utk), api._jvec(tsave),
            nl=nl, nw=nw, nt=nt, t_salt=t_salt_s, **dec,
            rest_t=float(rest.get("rest_t", 293.15)), rest_h=float(rest.get("rest_h", 1.5)),
            rest_eps=float(rest.get("rest_eps", 0.10)), rest_eta=float(rest.get("rest_eta", 0.9)), **extra,
        )
        p = jl.getproperty(s, jl.Symbol("piece"))
        get = lambda obj, name: np.array(jl.getproperty(obj, jl.Symbol(name)), copy=True)
        out = {n: get(p, n) for n in SERIES + FIELDS}
        out.update({n: float(get(p, n)) for n in SCALARS})
        for n in ("Tpit", "m_fuel"):
            out[n] = get(s, n)
        out["t_load"] = float(jl.getproperty(s, jl.Symbol("t_load")))
        out["mat"] = get(s, "mat").astype(int)
        out["bone"] = get(s, "bone")
        out["faces"] = get(s, "faces").astype(int).reshape(-1, 3) if shape is not None else None
        out["tris"] = get(s, "tris") if shape is not None else None          # the smooth surface (n × 9, m)
        out["tri_face"] = get(s, "tri_f").astype(int) - 1 if shape is not None else None   # 0-based; −1 none
        mesh = jl.getproperty(s, jl.Symbol("mesh"))
        out["coords"] = get(mesh, "coords")
        out["dims"] = tuple(int(d) for d in jl.getproperty(mesh, jl.Symbol("dims")))
        bnd = jl.getproperty(mesh, jl.Symbol("boundary"))
        out["boundary_nodes"] = np.array([int(b[0]) for b in bnd]) - 1        # 0-based
        out["boundary_areas"] = np.array([float(b[1]) for b in bnd])
        out["core"] = int(jl.getproperty(mesh, jl.Symbol("core"))) - 1
        out["vol"] = get(mesh, "vol")
        out["area"] = float(jl.getproperty(mesh, jl.Symbol("area")))
        geo = jl.getproperty(s, jl.Symbol("geometry"))
        out["geometry"] = {k: float(jl.getproperty(geo, jl.Symbol(k))) for k in DIMS}
        cfg = jl.MeatConfig()
        out["lstar_map"] = {k: float(jl.getproperty(cfg, jl.Symbol(k))) for k in ("L0_bark", "dL_B", "dL_S")}
        return out

    sim = api._on_julia_thread(_run)
    if shape is None:
        sim.update(grid=f"{nl}x{nw}x{nt}", shape=None)
    else:
        sim.update(grid=f"{coarsen * shape['h'] * 1000:.0f} mm cells", shape=shape, coarsen=coarsen)
    sim.update(t=tsave, plan=plan_name, decisions=dec)
    return sim


# --- the flat's fields on its node grid ------------------------------------------------------------------------------

def node_grid(sim, values):
    """A per-node vector as (nl, nw, nt): length from the thick end, width from the centre plane, height from the grate."""
    nl, nw, nt = sim["dims"]
    return np.asarray(values).reshape((nl, nw, nt), order="F")


def coords_grid(sim):
    """The node coordinates [cm] as three (nl, nw, nt) arrays: length, width (from the centre plane), height."""
    c = 100.0 * np.asarray(sim["coords"])
    return node_grid(sim, c[:, 0]), node_grid(sim, c[:, 1]), node_grid(sim, c[:, 2])


def field_at(sim, name, k):
    """A node field at save index k as (nl, nw, nt)."""
    return node_grid(sim, np.asarray(sim[name])[:, k])


def face_state(sim, name, k, face):
    """A surface state at save index k on one exposed face, on that face's node grid: 'top' and 'grate' (nl, nw),
    'side' (nl, nt), 'thick' and 'thin' (nw, nt)."""
    nl, nw, nt = sim["dims"]
    d = np.asarray(sim[name])[:, k]
    n_tb = 2 * nl * nw
    if face in ("top", "grate"):
        return d[(0 if face == "top" else 1):n_tb:2].reshape((nl, nw), order="F")
    if face == "side":
        return d[n_tb:n_tb + nl * nt].reshape((nl, nt), order="F")
    o = n_tb + nl * nt
    return d[o + (0 if face == "thick" else 1):o + 2 * nw * nt:2].reshape((nw, nt), order="F")


def mirror_width(a, axis=1, negate=False):
    """Reflect an array across the centre plane along `axis` (the half-width index 0 on the plane): the whole width.
    `negate` flips the sign of the mirrored half (the width coordinate itself)."""
    a = np.asarray(a)
    rev = np.flip(a, axis=axis)
    rev = -rev if negate else rev
    rest = np.take(a, range(1, a.shape[axis]), axis=axis)
    return np.concatenate([rev, rest], axis=axis)
