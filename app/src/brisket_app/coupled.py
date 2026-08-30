"""End-to-end differentiable coupled cook.

The single function consumed unchanged by Bayesian calibration (NumPyro)
and optimal control (Optax).

``coupled_cook`` is one solve of the monolith Tesseract (pit + meat +
humidity together, forward-mode derivatives). The two-Tesseract Picard
composition it replaced is in the git history (docs/monolith.md has the
measured comparison).
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
    """Pit vapor balance (docs/model.md, pit humidity), quasi-steady.

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


ICS_KEYS = (
    "tpit0",
    "t_amb",
    "m0",
    "t0_meat",
    "t_field0",
    "w_field0",
    "c_field0",
    "f_field0",
    "bark_b0",
    "bark_s0",
    "ring0",
)


def coupled_cook(
    theta,
    phi,
    u_knots,
    tx,
    grids: CookGrids,
    abstol: float = 1e-6,
    reltol: float = 1e-6,
    ics: dict | None = None,
    t_wrap=1.0e9,
    t_pull=1.0e9,
    rest: dict | None = None,
    humidity: Humidity | None = None,
    t_add=1.0e9,
    t_spritz=1.0e9,
    dt_spritz=3600.0,
    n_meat: float = 1.0,
    spritz_dose=0.05,
):
    """Simulate the coupled pit-brisket system in ONE solve, differentiably.

    The monolith Tesseract integrates pit, meat and humidity together (no
    exchange grid, no Picard truncation) and exposes forward-mode jvp / vjp
    / jacobian endpoints, so this is differentiable w.r.t. theta, phi,
    u_knots, t_wrap and t_pull through tesseract-jax in both modes.

    Parameters
    ----------
    theta, phi : array-like
        Meat (7) and pit (7) parameters; phi[6] = a0 carries the ignition
        state for receding-horizon handoff.
    u_knots : array-like
        Damper schedule on ``grids.u_tknots``.
    tx : Tesseract
        The monolith client (``brisket_app.clients.monolith_client``).
    grids : CookGrids
        Save grid (``tknots``) and damper knots.
    ics : dict, optional
        Initial conditions: any of ICS_KEYS (pit state, meat state fields,
        bark states).
    t_wrap, t_pull : float
        Wrap and pull times [s]; 1e9 = never.
    rest : dict, optional
        Rest-environment overrides (``rest_t/rest_h/rest_eps/rest_eta``).
    humidity : Humidity, optional
        Pit vapor-balance constants.
    t_add : float
        Mid-cook fuel addition (one chimney) [s]; 1e9 = none.
    t_spritz, dt_spritz : float
        Spritz policy: first spritz and interval [s]; t_spritz = 1e9 = none.
    n_meat : float
        Identical briskets sharing the pit.
    spritz_dose : float
        Water per spritz per exposed area [kg/m²].
    """
    ics = ics or {}
    humidity = humidity or Humidity()
    out = apply_tesseract(
        tx,
        {
            "theta": theta,
            "phi": phi,
            "u_knots": u_knots,
            "u_tknots": jnp.asarray(grids.u_tknots),
            "tsave": jnp.asarray(grids.tknots),
            "t_wrap": jnp.asarray(t_wrap),
            "t_pull": jnp.asarray(t_pull),
            "t_add": jnp.asarray(t_add),
            "t_spritz": jnp.asarray(t_spritz),
            "dt_spritz": jnp.asarray(dt_spritz),
            "spritz_dose": jnp.asarray(spritz_dose),
            "n_meat": float(n_meat),
            "abstol": abstol,
            "reltol": reltol,
            "afr": humidity.afr,
            "y_w": humidity.y_w,
            "pan_kg_s": humidity.pan_kg_s,
            "omega_amb": humidity.omega_amb,
            **{k: ics[k] for k in ICS_KEYS if k in ics},
            **(rest or {}),
        },
    )
    return out


# Default parameters (shared with the Julia test suite). D_eff is an *effective*
# moisture transport coefficient — above molecular-diffusion literature values
# because it lumps in pressure-driven juice expulsion/wicking, the mechanism
# that actually keeps a brisket surface wet through the stall.
THETA_DEFAULT = np.array(
    [7.0e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6]
)  # D_eff, k_m, h_c, L, k_mult, k_e, a_myosin, m_conv
# C_th, UA, r_max, dH_eff, u_leak, k_spread, a0 — a0=1 makes the ignition
# dynamics inert (exactly the pre-ignition-state model)
PHI_DEFAULT = np.array([15e3, 20.0, 3e-4, 1.2e7, 0.2, 3e-4, 1.0])
