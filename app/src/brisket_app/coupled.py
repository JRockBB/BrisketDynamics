"""End-to-end differentiable coupled cook.

The single function consumed unchanged by Bayesian calibration (NumPyro)
and optimal control (Optax).

Coupling scheme: whole-horizon Picard iteration, K=2 fixed passes, unrolled —
the model is *defined* as the unrolled map, so JAX reverse-mode through the
composed Tesseract VJPs is the exact gradient of what we simulate.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

HR = 3600.0


@dataclass(frozen=True)
class Humidity:
    """Pit vapor balance (docs/tier1_plan.md §3), quasi-steady.

    Air mass flow follows the draft, ``afr * r_max * g(u)``; the base
    vapor pressure is ambient plus combustion water plus the water pan,
    diluted by that flow; the meat's own vapor is added inside the meat
    Tesseract (inverse air flow ``imair_knots``).
    """

    afr: float = 20.0  # air-to-fuel ratio of the draft [kg air / kg fuel at r_max]
    y_w: float = 0.28  # combustion water yield [kg H2O / kg charcoal]
    pan_kg_s: float = 0.0  # water-pan evaporation [kg/s]
    omega_amb: float = 0.007  # ambient absolute humidity [kg/kg]
    p_atm: float = 101325.0  # [Pa]


@dataclass(frozen=True)
class CookGrids:
    """Shared time grids: exchange/save grid and damper-knot grid.

    The cook runs to ``t_end`` (serve); after ``t_pull`` the meat integrates
    under the rest boundary condition while the pit keeps burning.
    """

    t_end: float = 12 * HR
    dt_exchange: float = 300.0  # 5-min exchange/save grid
    n_u: int = 16  # damper schedule knots
    tknots: np.ndarray = field(init=False)
    u_tknots: np.ndarray = field(init=False)

    def __post_init__(self):
        """Materialize the derived exchange and damper-knot grids."""
        n = max(2, round(self.t_end / self.dt_exchange) + 1)
        object.__setattr__(self, "tknots", np.linspace(0.0, self.t_end, n))
        object.__setattr__(self, "u_tknots", np.linspace(0.0, self.t_end, self.n_u))

    @property
    def nt(self) -> int:
        """Number of exchange/save grid points."""
        return len(self.tknots)


def coupled_cook(
    theta,
    phi,
    u_knots,
    meat_tx,
    pit_tx,
    grids: CookGrids,
    K: int = 2,
    abstol: float = 1e-6,
    reltol: float = 1e-6,
    ics: dict | None = None,
    t_wrap=1.0e9,
    t_pull=1.0e9,
    rest: dict | None = None,
    humidity: Humidity | None = None,
):
    """Simulate the coupled pit-brisket system, differentiably.

    Differentiable w.r.t. theta (meat params), phi (pit params), u_knots
    (damper schedule), t_wrap (Texas-crutch wrap time [s]; default = never
    wrap), and t_pull (pull time [s]; default = never pull — the meat then
    rests under the ``rest`` environment, a dict of the meat Tesseract's
    plain ``rest_t/rest_h/rest_eps/rest_eta`` inputs; None = its cambro
    defaults). Quality outputs are evaluated at the end of the horizon
    (serve). theta's L is the raw half-thickness;
    ``l_thick`` tracks the physical one as the meat shrinks.

    Returns
    -------
    dict
        tpit, tcore, tsurf, j_evap, q_meat, m_fuel, c_field_final, bark and
        moisture outputs (bark_b/bark_s/l_star, juiciness, fat_rendered,
        evap_w/drip_w/drip_f), and the Picard residual diagnostic picard_dT
        (max |dT_pit| between the last two iterations; NaN when K=1).
    """
    # ics: optional initial/boundary conditions, e.g. for a real cook log:
    # {"tpit0": K, "t_amb": K, "m0": kg, "t0_meat": K}. Defaults (cold pit,
    # fridge meat) live in the Tesseract schemas.
    ics = ics or {}
    humidity = humidity or Humidity()
    pit_ics = {k: ics[k] for k in ("tpit0", "t_amb", "m0") if k in ics}
    meat_ics = {
        k: ics[k]
        for k in (
            "t0_meat",
            "t_field0",
            "w_field0",
            "c_field0",
            "f_field0",
            "bark_b0",
            "bark_s0",
        )
        if k in ics
    }
    tk = jnp.asarray(grids.tknots)
    utk = jnp.asarray(grids.u_tknots)
    q = jnp.zeros(grids.nt)
    tpit_prev = None
    pit = meat = None
    for _ in range(K):
        # optimization_barrier: each Tesseract call gets its own uncollapsed
        # copy of shared inputs. Without it, XLA reuses one buffer across the
        # K unrolled callback invocations, and the second consumption has been
        # observed to read recycled memory (all-NaN damper schedule at iter 1
        # under jit; flaky shape corruption in eager mode).
        phi_i, u_i, q_i = jax.lax.optimization_barrier((phi, u_knots, q))
        pit = apply_tesseract(
            pit_tx,
            {
                "phi": phi_i,
                "u_knots": u_i,
                "u_tknots": utk,
                "qmeat_knots": q_i,
                "q_tknots": tk,
                "tsave": tk,
                "abstol": min(abstol, 1e-8),
                "reltol": min(reltol, 1e-8),
                **pit_ics,
            },
        )
        # pit vapor balance on the exchange grid: draft from the damper
        # (the same piecewise-linear knots the pit integrates), base vapor
        # pressure from ambient + combustion water + pan
        u_tk = jnp.interp(tk, utk, u_knots)
        mair = humidity.afr * phi[2] * (phi[4] + (1.0 - phi[4]) * u_tk)
        pv_base = (humidity.p_atm / 0.622) * (
            humidity.omega_amb + (humidity.y_w * pit["mdot"] + humidity.pan_kg_s) / mair
        )
        theta_i, tpit_i, twrap_i, smoke_i, tpull_i, pvb_i, imair_i = (
            jax.lax.optimization_barrier(
                (
                    theta,
                    pit["tpit"],
                    jnp.asarray(t_wrap),
                    pit["mdot"],
                    jnp.asarray(t_pull),
                    pv_base,
                    1.0 / mair,
                )
            )
        )
        meat = apply_tesseract(
            meat_tx,
            {
                "theta": theta_i,
                "tpit_knots": tpit_i,
                "t_wrap": twrap_i,
                "smoke_knots": smoke_i,
                "t_pull": tpull_i,
                "pv_base_knots": pvb_i,
                "imair_knots": imair_i,
                "tknots": tk,
                "tsave": tk,
                "abstol": abstol,
                "reltol": reltol,
                **meat_ics,
                **(rest or {}),
            },
        )
        q = meat["q_meat"]
        picard = (
            jnp.max(jnp.abs(pit["tpit"] - tpit_prev))
            if tpit_prev is not None
            else jnp.nan
        )
        tpit_prev = pit["tpit"]

    return {
        "tpit": pit["tpit"],
        "m_fuel": pit["m_fuel"],
        "tcore": meat["tcore"],
        "tsurf": meat["tsurf"],
        "j_evap": meat["j_evap"],
        "q_meat": meat["q_meat"],
        "c_field_final": meat["c_field_final"],
        "w_retained": meat["w_retained"],
        "t_field_final": meat["t_field_final"],
        "w_field_final": meat["w_field_final"],
        "f_field_final": meat["f_field_final"],
        "ignited": pit["ignited"],
        "mdot": pit["mdot"],
        "bark_b": meat["bark_b"],
        "bark_s": meat["bark_s"],
        "l_star": meat["l_star"],
        "juiciness": meat["juiciness"],
        "fat_rendered": meat["fat_rendered"],
        "evap_w": meat["evap_w"],
        "drip_w": meat["drip_w"],
        "drip_f": meat["drip_f"],
        "l_thick": meat["l_thick"],
        "pv_pit": meat["pv_pit"],
        "picard_dT": picard,
    }


# Default parameters (shared with the Julia test suite). D_eff is an *effective*
# moisture transport coefficient — above molecular-diffusion literature values
# because it lumps in pressure-driven juice expulsion/wicking, the mechanism
# that actually keeps a brisket surface wet through the stall.
THETA_DEFAULT = np.array(
    [7.0e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20]
)  # D_eff, k_m, h_c, L, k_mult, k_e, a_myosin
# C_th, UA, r_max, dH_eff, u_leak, k_spread, a0 — a0=1 makes the ignition
# dynamics inert (exactly the pre-ignition-state model)
PHI_DEFAULT = np.array([15e3, 20.0, 3e-4, 1.2e7, 0.2, 3e-4, 1.0])

PICARD_WARN_K = 2.0


def check_picard(sim, log=print, threshold: float = PICARD_WARN_K):
    """Log the K-pass residual; warn when coupling truncation exceeds `threshold`."""
    d = float(np.asarray(sim.get("picard_dT", np.nan)))
    msg = f"Picard residual max|dT_pit| = {d:.3f} K"
    if np.isfinite(d) and d > threshold:
        log(f"WARNING: {msg} exceeds {threshold} K (K=2 coupling not converged)")
    else:
        log(msg)
    return d
