"""Gauss-Newton / Fisher machinery for the real cook on the monolith Tesseract.

`model_mean` returns the pit and core trajectories on the cook-log grid and
their Jacobian w.r.t. the 15 physics latents through the Tesseract's
``apply`` and ``jacobian`` endpoints (one coupled solve, forward mode);
`Whitened` turns the masked AR(1) likelihood into a least-squares residual so
Levenberg-Marquardt and the Fisher information apply directly. Parameters
live in numpyro's unconstrained space (log for LogNormal sites, logit for
Beta sites) so the Fisher covariance is comparable with the Laplace one.

A log that documents its preheat (``meta["preheat_s"] > 0``) is fitted from
lighting (`SITES_LIT`): the pit at ambient with a full fuel ring, the bed a
chimney start's ``A_LIGHT`` (a0 is no longer a latent), the unlogged preheat
damper one latent ``u_pre`` held over the preheat, the meat loaded at the
documented time (with its lid pulse); the observations are the log's, from
meat-on. A log without one keeps the start-of-log layout (`SITES`: the pit
from its first reading, a0 a latent).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpyro.distributions as dist

from .calibrate import (
    PHI_SITES,
    PHI_SITES_IGNITION,
    PREHEAT_SITES,  # noqa: F401  (re-exported: the from-lighting latent)
    SIGMA_FLOOR_K,
    SITES_LIT,
    THETA_SITES,
    contiguous_runs,
    from_lighting,
    lighting_grids,
)
from .clients import monolith_client
from .coupled import A_LIGHT

SITES = (
    THETA_SITES + PHI_SITES + PHI_SITES_IGNITION
)  # monolith order, 15 physics latents
NAMES = [n for n, _ in SITES]
N_PHYS = len(SITES)
N_THETA = len(THETA_SITES)
T_AMB = 300.0
NOISE_NAMES = ["sigma_pit", "sigma_core", "rho_pit", "rho_core"]
# from lighting (calibrate.SITES_LIT): a0 fixed at A_LIGHT, the preheat damper u_pre a vague latent
NAMES_LIT = [n for n, _ in SITES_LIT]
assert PHI_SITES_IGNITION[0][0] == "k_spread" and len(SITES_LIT) == len(SITES)


def sites_for(log: dict):
    """Return the fit's sites for this log: `SITES_LIT` from lighting, else `SITES`."""
    return SITES_LIT if from_lighting(log) else SITES


def names_for(log: dict) -> list[str]:
    """Return the fit's latent names for this log."""
    return [n for n, _ in sites_for(log)]


def phi_of(x, log: dict) -> np.ndarray:
    """Return the 7-vector phi of the fit's latents ``x`` (a0 = A_LIGHT from lighting)."""
    x = np.asarray(x, dtype=float)
    if from_lighting(log):
        return np.concatenate([x[N_THETA : N_THETA + len(PHI_SITES) + 1], [A_LIGHT]])
    return x[N_THETA:]
# numpyro flattens the unconstrained params dict in sorted-key order
ALL_NAMES_SORTED = sorted(NAMES + NOISE_NAMES)


def n_preheat(log: dict) -> int:
    """Grid points of the preheat before the log's t = 0 (the log's spacing; 0 without one)."""
    return lighting_grids(log)[1] if from_lighting(log) else 0


def _inputs(x, log: dict) -> dict:
    x = np.asarray(x, dtype=float)
    t = np.asarray(log["t"], dtype=float)
    u = np.asarray(log["u"], dtype=float)
    t_amb = float(log["meta"].get("t_amb_K", T_AMB))
    if not from_lighting(log):
        return {
            "theta": x[:N_THETA],
            "phi": x[N_THETA:],
            "u_knots": u,
            "u_tknots": t,
            "tsave": t,
            "tpit0": float(log["meta"]["tpit0_K"]),
            "t_amb": t_amb,
            "t0_meat": float(log["meta"]["tcore0_K"]),
        }
    # from lighting: the preheat's knots on the log's spacing (calibrate.lighting_grids, the NumPyro model's grid),
    # the damper u_pre over them, the meat on at the end
    grids, n = lighting_grids(log)
    tl = np.asarray(grids.tknots, dtype=float)
    return {
        "theta": x[:N_THETA],
        "phi": phi_of(x, log),
        "u_knots": np.concatenate([np.full(n, x[-1]), u]),
        "u_tknots": tl,
        "tsave": tl,
        "tpit0": t_amb,
        "t_amb": t_amb,
        "t0_meat": float(log["meta"]["tcore0_K"]),
        "t_load": np.asarray(float(tl[n])),
    }


def model_mean(x, log: dict, jac: bool = True):
    """μ = [T_pit(t); T_core(t)] on the log grid at physics params `x`, and dμ/dx.

    Parameters
    ----------
    x : array-like, length 15
        Physics latents in the log's order (`sites_for`: THETA_SITES then
        PHI_SITES + PHI_SITES_IGNITION, or from lighting k_spread + u_pre),
        constrained space.
    log : dict
        `cooklog.load` output.
    jac : bool
        Also return the (2·nt × 14) Jacobian (the Tesseract's ``jacobian``
        endpoint: one forward-mode pass over the 14 columns).

    Notes
    -----
    The Jacobian of an adaptive solve jitters at ~4e-5 relative under
    last-bit input changes at the 1e-6 tolerances; near an optimum the
    potential's gradient cancels O(1e3) terms, so that jitter reads as
    ~0.5 % there.
    """
    tx = monolith_client()
    inp = _inputs(x, log)
    n = n_preheat(log)                                                # the observations are from meat-on
    out = tx.apply(inp)
    mu = np.concatenate([np.asarray(out["tpit"])[n:], np.asarray(out["tcore"])[n:]])
    if not jac:
        return mu, None
    J = tx.jacobian(inp, ["theta", "phi"], ["tpit", "tcore"])
    if n == 0:
        rows = [
            np.hstack([np.asarray(J[k]["theta"]), np.asarray(J[k]["phi"])])
            for k in ("tpit", "tcore")
        ]
        return mu, np.vstack(rows)
    # from lighting: phi's a0 column is fixed (dropped); u_pre's column is the damper knots' over the preheat (one
    # directional derivative)
    du = np.zeros(len(inp["u_knots"]))
    du[:n] = 1.0
    jv = tx.jacobian_vector_product(inp, ["u_knots"], ["tpit", "tcore"], {"u_knots": du})
    rows = [
        np.hstack([
            np.asarray(J[k]["theta"])[n:],
            np.asarray(J[k]["phi"])[n:, :-1],
            np.asarray(jv[k])[n:, None],
        ])
        for k in ("tpit", "tcore")
    ]
    return mu, np.vstack(rows)


def preheat_trace(x, log: dict) -> dict:
    """Return the fitted cook's pit and core over the whole grid from lighting.

    The preheat included; times [s] from lighting, the log's t = 0 at ``t_load``.
    """
    inp = _inputs(x, log)
    out = monolith_client().apply(inp)
    return {"t": np.asarray(inp["tsave"]), "tpit": np.asarray(out["tpit"]), "tcore": np.asarray(out["tcore"]),
            "t_load": float(log["meta"].get("preheat_s", 0.0))}


# --- unconstrained parameterization and priors -------------------------------


def _is_beta(d) -> bool:
    return isinstance(d, dist.Beta)


def to_z(x, sites=SITES):
    """Constrained → unconstrained (log for LogNormal sites, logit for Beta sites)."""
    x = np.asarray(x, dtype=float)
    return np.array(
        [
            np.log(v / (1 - v)) if _is_beta(d) else np.log(v)
            for v, (_, d) in zip(x, sites, strict=True)
        ]
    )


def from_z(z, sites=SITES):
    """Unconstrained → constrained."""
    z = np.asarray(z, dtype=float)
    return np.array(
        [
            1 / (1 + np.exp(-v)) if _is_beta(d) else np.exp(v)
            for v, (_, d) in zip(z, sites, strict=True)
        ]
    )


def dx_dz(z, sites=SITES):
    """Diagonal of the transform Jacobian dx/dz."""
    x = from_z(z, sites)
    return np.array(
        [v * (1 - v) if _is_beta(d) else v for v, (_, d) in zip(x, sites, strict=True)]
    )


def prior_terms(z, sites=SITES):
    """−log p(z), its gradient and its (diagonal) Hessian in the unconstrained space.

    LogNormal sites are Gaussian in z; Beta(a, b) sites carry the logit
    Jacobian, so −log p(z) = −a·log σ(z) − b·log(1 − σ(z)).
    """
    z = np.asarray(z, dtype=float)
    f, g, h = 0.0, np.zeros_like(z), np.zeros_like(z)
    for i, (v, (_, d)) in enumerate(zip(z, sites, strict=True)):
        if _is_beta(d):
            a, b = float(d.concentration1), float(d.concentration0)
            s = 1 / (1 + np.exp(-v))
            f += -a * np.log(s) - b * np.log(1 - s)
            g[i] = -a * (1 - s) + b * s
            h[i] = (a + b) * s * (1 - s)
        else:
            m, sd = float(d.loc), float(d.scale)
            f += 0.5 * ((v - m) / sd) ** 2
            g[i] = (v - m) / sd**2
            h[i] = 1 / sd**2
    return f, g, h


# --- whitened AR(1) residual ---------------------------------------------------


@dataclass(frozen=True)
class Whitened:
    """The masked AR(1) likelihood of the cook log as a least-squares residual.

    ``w = L (y − μ)`` with ``L`` the per-run whitening (first point marginal,
    then innovations), so ``−log p(y | μ) = ½‖w‖² + const`` at fixed noise
    parameters; ``sigma_*`` are the effective marginal scales
    (SIGMA_FLOOR_K + site value).
    """

    L: np.ndarray  # (n_obs × 2·nt)
    y: np.ndarray  # (2·nt) stacked [pit; core] observations (unmasked entries unused)

    @classmethod
    def build(cls, log: dict, sigma_pit, rho_pit, sigma_core, rho_core):
        """Assemble the whitening operator from the log masks and noise parameters."""
        nt = len(log["t"])
        rows = []
        for off, mask, sig, rho in (
            (0, log["mask_pit"], sigma_pit, rho_pit),
            (nt, log["mask_core"], sigma_core, rho_core),
        ):
            innov = sig * np.sqrt(max(1.0 - rho * rho, 1e-8))
            for run in contiguous_runs(np.asarray(mask)):
                r = np.zeros(2 * nt)
                r[off + run[0]] = 1.0 / sig
                rows.append(r)
                for k in range(1, len(run)):
                    r = np.zeros(2 * nt)
                    r[off + run[k]] = 1.0 / innov
                    r[off + run[k - 1]] = -rho / innov
                    rows.append(r)
        y = np.concatenate([np.asarray(log["obs_pit"]), np.asarray(log["obs_core"])])
        return cls(L=np.array(rows), y=y)

    def residual(self, mu):
        """Whitened residual w = L (y − μ)."""
        return self.L @ (self.y - mu)

    def jacobian(self, J_mu):
        """Jacobian of w with respect to the model-mean parameters: −L J."""
        return -self.L @ J_mu


def effective_sigma(params: dict) -> tuple[float, float, float, float]:
    """(sigma_pit, rho_pit, sigma_core, rho_core) with the digitization floor applied."""
    return (
        SIGMA_FLOOR_K + float(params["sigma_pit"]),
        float(params["rho_pit"]),
        SIGMA_FLOOR_K + float(params["sigma_core"]),
        float(params["rho_core"]),
    )


# --- Levenberg-Marquardt ------------------------------------------------------


def objective(z, wh: Whitened, log: dict, jac: bool = True):
    """F(z) = ½‖w‖² − log p(z) (up to constants), with gradient and Gauss-Newton Hessian."""
    sites = sites_for(log)
    mu, J = model_mean(from_z(z, sites), log, jac=jac)
    w = wh.residual(mu)
    fp, gp, hp = prior_terms(z, sites)
    F = 0.5 * float(w @ w) + fp
    if not jac:
        return F, None, None, w
    Jw = wh.jacobian(J) * dx_dz(z, sites)[None, :]
    g = Jw.T @ w + gp
    H = Jw.T @ Jw + np.diag(hp)
    return F, g, H, w


def levenberg_marquardt(
    z0,
    wh: Whitened,
    log: dict,
    iters: int = 30,
    lam: float = 1e-2,
    tol: float = 1e-4,
    verbose=print,
):
    """Minimize the whitened objective from `z0`; returns (z, F, H, history)."""
    z = np.asarray(z0, dtype=float)
    F, g, H, w = objective(z, wh, log)
    hist = [(F, float(np.linalg.norm(g)))]
    verbose(
        f"  it  0  F={F:10.3f}  |g|={np.linalg.norm(g):9.3g}  0.5|w|^2={0.5 * w @ w:9.3f}"
    )
    for it in range(1, iters + 1):
        while True:
            A = H + lam * np.diag(np.diag(H))
            step = -np.linalg.solve(A, g)
            F_try, *_ = objective(z + step, wh, log, jac=False)
            if F_try < F:
                lam = max(lam * 0.3, 1e-8)
                break
            lam *= 10.0
            if lam > 1e10:
                verbose("  LM: no descent direction - stopping")
                return z, F, H, hist
        z = z + step
        F, g, H, w = objective(z, wh, log)
        hist.append((F, float(np.linalg.norm(g))))
        verbose(
            f"  it {it:2d}  F={F:10.3f}  |g|={np.linalg.norm(g):9.3g}  "
            f"0.5|w|^2={0.5 * w @ w:9.3f}  |step|={np.linalg.norm(step):.3g}  lam={lam:.1e}"
        )
        if np.linalg.norm(step) < tol:
            break
    return z, F, H, hist


def fisher_inverse_mass(x_map, log: dict, noise: dict, hess_npz=None) -> np.ndarray:
    """Dense inverse mass matrix from the Fisher information at the MAP.

    The 15 physics coordinates get the Gauss-Newton covariance
    (Jᵀ Σ⁻¹ J + prior curvature)⁻¹ in the unconstrained space — ridge widths
    included, no eigenvalue floor; the 4 noise coordinates get the
    noise block of run_laplace.py's Gauss-Newton Hessian (exact in the
    noise coordinates) when ``hess_npz`` is given, else the identity. Assembled in numpyro's
    sorted-site order with no cross terms between the two groups.
    """
    wh = Whitened.build(log, *effective_sigma(noise))
    _F, _g, H, _w = objective(to_z(x_map), wh, log)
    cov_phys = np.linalg.inv(H)
    n = len(ALL_NAMES_SORTED)
    M = np.zeros((n, n))
    idx = [ALL_NAMES_SORTED.index(k) for k in NAMES]
    M[np.ix_(idx, idx)] = cov_phys
    nidx = [ALL_NAMES_SORTED.index(k) for k in NOISE_NAMES]
    if hess_npz is not None:
        c = np.load(hess_npz)
        Hl = 0.5 * (c["H"] + c["H"].T)
        Hn = Hl[np.ix_(nidx, nidx)]
        lam = np.linalg.eigvalsh(Hn)
        if lam.min() <= 0:
            Hn = Hn + (1e-3 * lam.max() - lam.min()) * np.eye(len(nidx))
        M[np.ix_(nidx, nidx)] = np.linalg.inv(Hn)
    else:
        M[np.ix_(nidx, nidx)] = np.eye(len(nidx))
    return M


# Physics rest in monolith (NAMES) order: everything except ignition.
_IGN_IN_PHYS = [NAMES.index(n) for n in ("a0", "k_spread")]


def sorted_to_phys(z_sorted: np.ndarray) -> np.ndarray:
    """ALL_NAMES_SORTED unconstrained vector -> NAMES unconstrained vector."""
    z_sorted = np.asarray(z_sorted, dtype=float).reshape(-1)
    return np.array([z_sorted[ALL_NAMES_SORTED.index(n)] for n in NAMES])


def apply_phys(z_sorted: np.ndarray, z_phys: np.ndarray) -> np.ndarray:
    """Write NAMES unconstrained values back into a sorted-site vector."""
    out = np.asarray(z_sorted, dtype=float).copy()
    z_phys = np.asarray(z_phys, dtype=float).reshape(-1)
    for i, n in enumerate(NAMES):
        out[ALL_NAMES_SORTED.index(n)] = z_phys[i]
    return out


def gn_block_newton(
    z_sorted,
    log: dict,
    wh: Whitened,
    hold_phys: list[int] | None = None,
    clip: float = 0.3,
    n_steps: int = 1,
    tol: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Gauss-Newton steps on physics not in ``hold_phys`` (NAMES indices).

    Local Fisher at the current rest, not the MAP Hessian. With ``tol > 0``
    the loop runs until ``|dz|_∞ < tol`` (up to ``n_steps``, now a cap) and
    the converged step is not taken — the TMB/INLA marginal formula
    ``−U(θ, û) − ½ log det H_rr`` assumes the inner optimum was reached, so
    an unconverged cell biases the marginal low exactly in the tails.
    ``tol = 0`` keeps the fixed step count. Returns
    ``(z_new, log det H_rr, |dz| of the last computed step, H_rr)``.
    """
    hold = set(hold_phys if hold_phys is not None else _IGN_IN_PHYS)
    rest = [i for i in range(N_PHYS) if i not in hold]
    z_sorted = np.asarray(z_sorted, dtype=float).copy()
    logdet = 0.0
    step_norm = 0.0
    eye = np.eye(len(rest))
    hrr = eye.copy()
    for _ in range(int(n_steps)):
        z_phys = sorted_to_phys(z_sorted)
        _f, g, h, _w = objective(z_phys, wh, log, jac=True)
        hrr = h[np.ix_(rest, rest)]
        hrr = 0.5 * (hrr + hrr.T) + 1e-6 * eye
        dz = np.clip(-np.linalg.solve(hrr, g[rest]), -clip, clip)
        _sign, logdet = np.linalg.slogdet(hrr)
        step_norm = float(np.linalg.norm(dz))
        if tol > 0.0 and float(np.abs(dz).max()) < tol:
            break
        z_phys_rest = z_phys.copy()
        z_phys_rest[rest] = z_phys[rest] + dz
        z_sorted = apply_phys(z_sorted, z_phys_rest)
    return z_sorted, float(logdet), step_norm, hrr


def gn_rest_newton(
    z_sorted,
    log: dict,
    wh: Whitened,
    clip: float = 0.3,
    n_steps: int = 1,
    tol: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Gauss-Newton Newton steps on non-ignition physics, ignition held fixed."""
    return gn_block_newton(
        z_sorted, log, wh, hold_phys=_IGN_IN_PHYS, clip=clip, n_steps=n_steps, tol=tol
    )
