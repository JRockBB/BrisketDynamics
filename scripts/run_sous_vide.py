"""Sous vide against Baldwin's heating tables: the data-free validation.

The generalization claim of docs/roadmap.md item "meats and cookers" needs
at least one case validated without new instruments. Sous vide is it: the
bath is the rest environment from t = 0 (eta_rest = 1 kills evaporation
exactly, eps_rest = 0, h_rest = the bath film coefficient), the geometry
mode supplies slab / cylinder / sphere, and Douglas Baldwin's published
heating-time table ("A Practical Guide to Sous Vide Cooking", Table 2.2)
is the external reference: time to heat from 5 C to within 0.5 C of the
bath, per thickness and shape, computed by him from conduction with
alpha = 1.4e-7 m2/s and h = 95 W/m2K and checked against real cuts.

Our model is NOT pure conduction — Choi-Okos properties of the local
composition and temperature, plus the water physics — yet it matches the
exact first-term conduction series at Baldwin's own alpha/h to <= 8% on
all 21 cases (mean ratio 1.01-1.03 per shape). Baldwin's published
cylinder and sphere columns sit ABOVE that exact solution (safety
padding, ~+15% and ~+35%); the slab column matches it, and us, to ~1%.

Run from app/:  uv run python -u ../scripts/run_sous_vide.py
Writes data/validation/sous_vide_baldwin.json and
figures/validation/sous_vide_baldwin.png.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "app" / "src"))

# Baldwin Table 2.2 (douglasbaldwin.com/sous-vide.html): minutes to heat
# from 5 C to within 0.5 C of the bath. Thickness = full slab thickness /
# cylinder or sphere diameter [mm].
BALDWIN_MIN = {
    "slab": {10: 19, 20: 50, 25: 75, 30: 90, 40: 150, 50: 210, 60: 285},
    "cylinder": {10: 11, 20: 30, 25: 40, 30: 50, 40: 75, 50: 120, 60: 150},
    "sphere": {10: 8, 20: 20, 25: 25, 30: 35, 40: 55, 50: 90, 60: 120},
}
GEOM_M = {"slab": 0.0, "cylinder": 1.0, "sphere": 2.0}
T0_K = 278.15  # 5 C
T_BATH_K = 333.15  # 60 C bath (the table is quoted as bath-independent)
H_BATH = 95.0  # Baldwin's surface coefficient [W/m2K]
DT_DONE = 0.5  # "heated through" = within this of the bath [K]
ALPHA_B = 1.4e-7  # Baldwin's thermal diffusivity [m2/s]
K_B = 0.48  # conductivity consistent with meat at that alpha [W/mK] (for Bi)


def theory_minutes(shape: str, L: float) -> float:
    """First-term exact conduction series at Baldwin's alpha/h.

    Time for the center of a slab/cylinder/sphere (half-thickness or
    radius L) to reach within DT_DONE of the bath, convective BC with
    Biot = h L / k: theta0 = A1 exp(-lambda1^2 Fo).
    """
    from scipy.optimize import brentq
    from scipy.special import j0, j1

    bi = H_BATH * L / K_B
    if shape == "slab":
        lam = brentq(lambda x: x * np.tan(x) - bi, 1e-9, np.pi / 2 - 1e-9)
        a1 = 2 * np.sin(lam) / (lam + np.sin(lam) * np.cos(lam))
    elif shape == "cylinder":
        lam = brentq(lambda x: x * j1(x) / j0(x) - bi, 1e-9, 2.4048 - 1e-9)
        a1 = (2 / lam) * j1(lam) / (j0(lam) ** 2 + j1(lam) ** 2)
    else:  # sphere
        lam = brentq(lambda x: 1 - x / np.tan(x) - bi, 1e-9, np.pi - 1e-9)
        a1 = 2 * (np.sin(lam) - lam * np.cos(lam)) / (lam - np.sin(lam) * np.cos(lam))
    theta0 = DT_DONE / (T_BATH_K - T0_K)
    fo = np.log(a1 / theta0) / lam**2
    return float(fo * L**2 / ALPHA_B / 60.0)


def main():
    """Sweep Baldwin's table through the bath-configured meat solver."""
    from juliacall import Main as jl

    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO}"; io=devnull)')
    jl.seval("using BrisketDynamics: meat_solve, MeatConfig")
    print("Julia ready", flush=True)

    theta = [3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6]
    rows = []
    t0 = time.perf_counter()
    for shape, table in BALDWIN_MIN.items():
        for d_mm, mins in table.items():
            L = d_mm / 2 / 1000.0  # half-thickness / radius [m]
            th = list(theta)
            th[3] = L
            t_end = max(4.0 * mins * 60.0, 3600.0)
            tsave = np.arange(0.0, t_end + 30.0, 30.0)
            cfg = jl.MeatConfig(
                T0=T0_K,
                geom_m=GEOM_M[shape],
                T_rest=T_BATH_K,
                h_rest=H_BATH,
                ε_rest=0.0,
                η_rest=1.0,
            )
            tpit = np.full(len(tsave), T_BATH_K)
            out = jl.meat_solve(
                jl.Vector(th),
                jl.Vector(tpit),
                jl.Vector(tsave),
                jl.Vector(tsave),
                cfg=cfg,
                t_pull=-3000.0,
            )
            tc = np.asarray(out.Tcore).copy()
            hit = np.flatnonzero(tc >= T_BATH_K - DT_DONE)
            t_model_min = float(tsave[hit[0]] / 60.0) if hit.size else float("nan")
            t_theory = theory_minutes(shape, L)
            rows.append(
                {
                    "shape": shape,
                    "thickness_mm": d_mm,
                    "baldwin_min": mins,
                    "theory_min": t_theory,
                    "model_min": t_model_min,
                    "ratio_baldwin": t_model_min / mins if hit.size else None,
                    "ratio_theory": t_model_min / t_theory if hit.size else None,
                }
            )
            print(
                f"  {shape:8s} {d_mm:3d} mm  model {t_model_min:6.1f}  "
                f"theory {t_theory:6.1f}  Baldwin {mins:4d} min   "
                f"model/theory {t_model_min / t_theory:5.2f}  "
                f"model/Baldwin {t_model_min / mins:5.2f}",
                flush=True,
            )
    wall = time.perf_counter() - t0

    summary = {}
    for s in BALDWIN_MIN:
        rt = [r["ratio_theory"] for r in rows if r["shape"] == s and r["ratio_theory"]]
        rb = [r["ratio_baldwin"] for r in rows if r["shape"] == s and r["ratio_baldwin"]]
        summary[s] = {
            "mean_ratio_theory": float(np.mean(rt)),
            "worst_abs_rel_theory": float(np.max(np.abs(np.array(rt) - 1.0))),
            "mean_ratio_baldwin": float(np.mean(rb)),
        }
        print(
            f"{s:8s} model/theory mean {summary[s]['mean_ratio_theory']:.3f} "
            f"(worst |rel| {summary[s]['worst_abs_rel_theory']:.2f})   "
            f"model/Baldwin mean {summary[s]['mean_ratio_baldwin']:.2f}",
            flush=True,
        )

    out_dir = REPO / "data" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference": "Baldwin, A Practical Guide to Sous Vide Cooking, "
        "Table 2.2 (douglasbaldwin.com); alpha=1.4e-7 m2/s, h=95 W/m2K, "
        "5C start, done = bath - 0.5C",
        "bath_c": T_BATH_K - 273.15,
        "h_bath": H_BATH,
        "note": "model is full physics (Choi-Okos properties, water "
        "transport), not Baldwin's constant-alpha conduction; theory_min = "
        "first-term exact conduction series at Baldwin's alpha/h (k=0.48 "
        "for Bi). Baldwin's published cylinder/sphere columns are "
        "safety-padded relative to that exact solution",
        "alpha_theory": ALPHA_B,
        "k_theory": K_B,
        "wall_s": wall,
        "rows": rows,
        "summary": summary,
    }
    (out_dir / "sous_vide_baldwin.json").write_text(json.dumps(payload, indent=2))
    print("wrote data/validation/sous_vide_baldwin.json")
    _figure(payload, REPO)


def _figure(payload: dict, repo: Path):
    """Model vs Baldwin heating times per geometry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"slab": "tab:blue", "cylinder": "tab:green", "sphere": "tab:orange"}
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for shape, color in colors.items():
        rows = [r for r in payload["rows"] if r["shape"] == shape]
        d = [r["thickness_mm"] for r in rows]
        ax.plot(
            d,
            [r["baldwin_min"] for r in rows],
            "o--",
            color=color,
            ms=4,
            alpha=0.55,
            label=f"{shape} — Baldwin table (safety-padded)",
        )
        ax.plot(
            d,
            [r["theory_min"] for r in rows],
            ":",
            color=color,
            lw=1.4,
            label=f"{shape} — exact conduction",
        )
        ax.plot(
            d,
            [r["model_min"] for r in rows],
            "s-",
            color=color,
            ms=4,
            label=f"{shape} — model",
        )
    ax.set(
        xlabel="thickness / diameter [mm]",
        ylabel="time to bath − 0.5 °C [min]",
        title="Sous vide vs Baldwin's heating table (no fitted parameters)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(repo / "figures" / "validation" / "sous_vide_baldwin.png", dpi=140)
    plt.close(fig)
    print("wrote figures/validation/sous_vide_baldwin.png")


if __name__ == "__main__":
    main()
