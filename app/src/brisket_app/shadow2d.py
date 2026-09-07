"""2-D shadow scoring: full-fidelity verification of 1-D plans.

The planners stay on the 1-D monolith; this module re-simulates a planned
cook on the 2-D axisymmetric Tesseract (shrink, salt, spritz, wrap, grain,
point-over-flat) and reports what only the grid can see — the TRUE cold
spot (the grid minimum, not the symmetric slab's center node), the bark
distribution over the exposed faces, and the current (shrunk) dimensions.
"""

import numpy as np

from .clients import slab2d_client
from .control import ControlConfig, lethality_dwell

# theta -> theta2: [D_eff, k_m, h_c, k_mult, k_e, a_myosin] (geometry is
# radius/height; m_conv acts through the schedule the plan already fixed)
THETA2_IDX = (0, 1, 2, 4, 5, 6)


def shadow_report(
    theta,
    phi,
    u_knots,
    grids,
    t_wrap: float = 1.0e9,
    t_salt: float = 0.0,
    t_spritz: float = 1.0e9,
    dt_spritz: float = 3600.0,
    cc: ControlConfig | None = None,
    radius: float = 0.10,
    nr: int = 16,
    nz: int = 14,
    h_step: float = 30.0,
    n_meat: float = 1.0,
    tx=None,
    **extra,
):
    """Re-simulate a planned cook in 2-D and report the grid-only quantities.

    Parameters
    ----------
    theta, phi : array-like
        The 1-D parameter vectors (theta 8; its L becomes height/2).
    u_knots : array-like
        The planned damper schedule on ``grids.u_tknots``.
    grids : CookGrids
        The plan's time discretization.
    t_wrap, t_salt, t_spritz, dt_spritz : float
        The planned decisions [s]; 1e9 = never, t_salt 0 = unsalted.
    cc : ControlConfig, optional
        Safety thresholds for the lethality/dwell report.
    radius, nr, nz, h_step, n_meat : float / int
        2-D surrogate geometry and resolution.
    tx : Tesseract, optional
        The slab2d client (defaults to the process-wide one).
    **extra
        Extra slab2d inputs (k_grain, f_top, tpit0, ...).

    Returns
    -------
    dict
        Cold-spot and core lethality/dwell (raw + pass/fail at the cc
        thresholds), retention, C_min, bark mean/spread, shrunk
        dimensions, and the final temperatures.
    """
    theta = np.asarray(theta, dtype=float)
    tx = tx or slab2d_client()
    cc = cc or ControlConfig()
    out = tx.apply(
        {
            "theta2": theta[list(THETA2_IDX)],
            "phi": np.asarray(phi, dtype=float),
            "u_knots": np.asarray(u_knots, dtype=float),
            "u_tknots": np.asarray(grids.u_tknots, dtype=float),
            "tsave": np.asarray(grids.tknots, dtype=float),
            "t_wrap": float(t_wrap),
            "t_salt": float(t_salt),
            "t_spritz": float(t_spritz),
            "dt_spritz": float(dt_spritz),
            "radius": float(radius),
            "height": 2.0 * float(theta[3]),
            "nr": int(nr),
            "nz": int(nz),
            "h_step": float(h_step),
            "n_meat": float(n_meat),
            **extra,
        }
    )
    t = np.asarray(grids.tknots, dtype=float)
    tcold = np.asarray(out["tcold"], dtype=float)
    tcore = np.asarray(out["tcore"], dtype=float)
    F_cold, d_cold = lethality_dwell(tcold, t, cc)
    F_core, d_core = lethality_dwell(tcore, t, cc)
    return {
        "F_cold": float(F_cold),
        "danger_h_cold": float(d_cold),
        "F_core": float(F_core),
        "danger_h_core": float(d_core),
        "cold_ok": bool(F_cold >= cc.leth_f_req and d_cold <= cc.danger_cap_h),
        "tcold_final_C": float(tcold[-1] - 273.15),
        "tcore_final_C": float(tcore[-1] - 273.15),
        "cold_gap_K": float(tcore[-1] - tcold[-1]),
        "w_retained": float(out["w_retained"]),
        "c_min": float(out["c_min"]),
        "bark_mean": float(out["bark_mean"]),
        "bark_std": float(out["bark_std"]),
        "salt_mean": float(out["salt_mean"]),
        "r_eff_cm": float(out["r_eff"]) * 100.0,
        "hz_eff_cm": float(out["hz_eff"]) * 100.0,
        "assumption": (
            "axisymmetric surrogate (radius set by hand, height = 2L); "
            "1-D-calibrated constants, bark distribution uncalibrated (C2)"
        ),
    }
