"""2-D shadow scoring: full-fidelity verification of 1-D plans.

The planners stay on the 1-D monolith; this module re-simulates a planned
cook on the 2-D axisymmetric Tesseract (shrink, salt, spritz, wrap, PULL
and rest, grain, point-over-flat) and reports what only the grid can see —
the TRUE cold spot (the grid minimum, not the symmetric slab's center
node), the bark distribution over the exposed faces, and the current
(shrunk) dimensions — at SERVE, the same moment the 1-D plan is scored:
after the planned pull the surrogate rests in the plan's rest environment
while the pit runs on, instead of cooking to the horizon.

Geometry. The surrogate is a finite cylinder of height 2L (the 1-D
half-thickness doubled) and, by default, the radius whose two end faces
carry the exposed area the 1-D coupling already assumes for one cut
(`MeatConfig.A_meat`, both faces of a flat): R = sqrt(A_meat / 2π), about
13.8 cm for the 0.12 m² default. Pass `radius` to override; the report
records which was used. Discretization: `refine=True` re-runs on a
doubled grid and reports the change in the serve cold spot, its peak,
and C_min, so a coarse grid cannot hide behind a headline number.
"""

import numpy as np

from .clients import slab2d_client
from .control import ControlConfig, lethality_cut

# theta -> theta2: [D_eff, k_m, h_c, k_mult, k_e, a_myosin] (geometry is
# radius/height; m_conv acts through the schedule the plan already fixed)
THETA2_IDX = (0, 1, 2, 4, 5, 6)
# MeatConfig.A_meat: one cut's exposed area in the 1-D pit coupling [m²]
A_MEAT_DEFAULT = 0.12


def equivalent_radius(a_meat: float = A_MEAT_DEFAULT) -> float:
    """Radius of the cylinder whose two end faces carry the 1-D exposed area."""
    return float(np.sqrt(a_meat / (2.0 * np.pi)))


def _apply(tx, theta, phi, u_knots, grids, decisions, radius, nr, nz, h_step,
           n_meat, rest, extra):
    return tx.apply(
        {
            "theta2": theta[list(THETA2_IDX)],
            "phi": np.asarray(phi, dtype=float),
            "u_knots": np.asarray(u_knots, dtype=float),
            "u_tknots": np.asarray(grids.u_tknots, dtype=float),
            "tsave": np.asarray(grids.tknots, dtype=float),
            **{k: float(v) for k, v in decisions.items()},
            "radius": float(radius),
            "height": 2.0 * float(theta[3]),
            "nr": int(nr),
            "nz": int(nz),
            "h_step": float(h_step),
            "n_meat": float(n_meat),
            **{k: float(v) for k, v in (rest or {}).items()},
            **extra,
        }
    )


def shadow_report(
    theta,
    phi,
    u_knots,
    grids,
    t_wrap: float = 1.0e9,
    t_salt: float = 0.0,
    t_spritz: float = 1.0e9,
    dt_spritz: float = 3600.0,
    t_pull: float = 1.0e9,
    rest: dict | None = None,
    cc: ControlConfig | None = None,
    radius: float | None = None,
    nr: int = 16,
    nz: int = 14,
    h_step: float = 30.0,
    n_meat: float = 1.0,
    refine: bool = False,
    tx=None,
    **extra,
):
    """Re-simulate a planned cook in 2-D to serve and report the grid-only quantities.

    Parameters
    ----------
    theta, phi : array-like
        The 1-D parameter vectors (theta 8; its L becomes height/2).
    u_knots : array-like
        The planned damper schedule on ``grids.u_tknots``.
    grids : CookGrids
        The plan's time discretization; ``grids.t_end`` is serve.
    t_wrap, t_salt, t_spritz, dt_spritz, t_pull : float
        The planned decisions [s]; 1e9 = never, t_salt 0 = unsalted. After
        ``t_pull`` the cut rests to serve (the 1-D pull blend on every face).
    rest : dict, optional
        The plan's rest environment (``rest_t``, ``rest_h``, ``rest_eps``,
        ``rest_eta``), the same keys the monolith takes; default cambro.
    cc : ControlConfig, optional
        Safety thresholds for the lethality / come-up-time report.
    radius : float, optional
        Cylinder radius [m]; default ``equivalent_radius()`` (the 1-D
        exposed area). The report records the source.
    nr, nz, h_step, n_meat : int / float
        Grid and step; identical cuts sharing the pit.
    refine : bool
        Re-run on a (2nr, 2nz) grid and report the discretization change.
    tx : Tesseract, optional
        The slab2d client (defaults to the process-wide one).
    **extra
        Extra slab2d inputs (k_grain, f_top, tpit0, t_amb, ...).

    Returns
    -------
    dict
        Cold-spot and core lethality / come-up time (raw + pass/fail at the
        cc thresholds), serve temperatures and the cold-spot peak,
        retention, C_min, bark mean/spread, shrunk dimensions, the geometry
        used and, with ``refine``, the grid-doubling deltas.
    """
    theta = np.asarray(theta, dtype=float)
    tx = tx or slab2d_client()
    cc = cc or ControlConfig()
    if radius is None:
        radius, radius_source = equivalent_radius(), "equivalent exposed area (A_meat)"
    else:
        radius, radius_source = float(radius), "given"
    decisions = {
        "t_wrap": t_wrap, "t_salt": t_salt, "t_spritz": t_spritz,
        "dt_spritz": dt_spritz, "t_pull": t_pull,
    }
    out = _apply(tx, theta, phi, u_knots, grids, decisions, radius, nr, nz,
                 h_step, n_meat, rest, extra)
    t = np.asarray(grids.tknots, dtype=float)
    tcold = np.asarray(out["tcold"], dtype=float)
    tcore = np.asarray(out["tcore"], dtype=float)
    F_cold, cut_cold = lethality_cut(tcold, t, cc)
    F_core, cut_core = lethality_cut(tcore, t, cc)
    cut_binding = bool(cc.cut_critical)
    rep = {
        "F_cold": float(F_cold),
        "cut_h_cold": float(cut_cold),
        "F_core": float(F_core),
        "cut_h_core": float(cut_core),
        "cut_critical": cut_binding,
        "cold_ok": bool(
            F_cold >= cc.leth_f_req and (cut_cold <= cc.cut_cap_h or not cut_binding)
        ),
        "t_pull_h": (float(t_pull) / 3600.0) if float(t_pull) < 1.0e8 else None,
        "tcold_final_C": float(tcold[-1] - 273.15),
        "tcold_peak_C": float(np.max(tcold) - 273.15),
        "tcore_final_C": float(tcore[-1] - 273.15),
        "cold_gap_K": float(tcore[-1] - tcold[-1]),
        "w_retained": float(out["w_retained"]),
        "c_min": float(out["c_min"]),
        "bark_mean": float(out["bark_mean"]),
        "bark_std": float(out["bark_std"]),
        "salt_mean": float(out["salt_mean"]),
        "r_eff_cm": float(out["r_eff"]) * 100.0,
        "hz_eff_cm": float(out["hz_eff"]) * 100.0,
        "radius_m": float(radius),
        "radius_source": radius_source,
        "height_m": 2.0 * float(theta[3]),
        "grid": f"{int(nr)}x{int(nz)}",
    }
    if refine:
        fine = _apply(tx, theta, phi, u_knots, grids, decisions, radius,
                      2 * nr, 2 * nz, h_step, n_meat, rest, extra)
        tcold_f = np.asarray(fine["tcold"], dtype=float)
        rep.update(
            {
                "grid_fine": f"{2 * int(nr)}x{2 * int(nz)}",
                "dtcold_serve_K": float(tcold_f[-1] - tcold[-1]),
                "dtcold_peak_K": float(np.max(tcold_f) - np.max(tcold)),
                "dc_min": float(fine["c_min"]) - float(out["c_min"]),
                "dw_retained": float(fine["w_retained"]) - float(out["w_retained"]),
            }
        )
    rep["assumption"] = (
        "axisymmetric surrogate (height = 2L; radius " + radius_source + "), "
        "rested to serve after the planned pull; 1-D-calibrated constants, "
        "bark distribution uncalibrated (C2); come-up time reported, "
        + ("binding" if cut_binding else "not binding for brisket (FSIS Gap 6)")
        + "; Appendix A humidity not modelled"
    )
    return rep
