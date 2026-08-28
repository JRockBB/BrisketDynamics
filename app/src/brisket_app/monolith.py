"""Gauss-Newton / Fisher machinery for the real cook on the monolith Tesseract.

`model_mean` returns the pit and core trajectories on the cook-log grid and
their Jacobian w.r.t. the 14 physics latents through the Tesseract's
``apply`` and ``jacobian`` endpoints (one coupled solve, forward mode);
`Whitened` turns the masked AR(1) likelihood into a least-squares residual so
Levenberg-Marquardt and the Fisher information apply directly. Parameters
live in numpyro's unconstrained space (log for LogNormal sites, logit for
Beta sites) so the Fisher covariance is comparable with the Laplace one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpyro.distributions as dist

from .calibrate import (
    PHI_SITES,
    PHI_SITES_IGNITION,
    SIGMA_FLOOR_K,
    THETA_SITES,
    contiguous_runs,
)
from .clients import monolith_client

SITES = (
    THETA_SITES + PHI_SITES + PHI_SITES_IGNITION
)  # monolith order, 14 physics latents
NAMES = [n for n, _ in SITES]
N_PHYS = len(SITES)
T_AMB = 300.0
NOISE_NAMES = ["sigma_pit", "sigma_core", "rho_pit", "rho_core"]
# numpyro flattens the unconstrained params dict in sorted-key order
ALL_NAMES_SORTED = sorted(NAMES + NOISE_NAMES)


def _inputs(x, log: dict) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "theta": x[:7],
        "phi": x[7:],
        "u_knots": np.asarray(log["u"], dtype=float),
        "u_tknots": np.asarray(log["t"], dtype=float),
        "tsave": np.asarray(log["t"], dtype=float),
        "tpit0": float(log["meta"]["tpit0_K"]),
        "t_amb": T_AMB,
        "t0_meat": float(log["meta"]["tcore0_K"]),
    }


def model_mean(x, log: dict, jac: bool = True):
    """μ = [T_pit(t); T_core(t)] on the log grid at physics params `x`, and dμ/dx.

    Parameters
    ----------
    x : array-like, length 14
        Physics latents in monolith order (THETA_SITES then PHI_SITES +
        PHI_SITES_IGNITION), constrained space.
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
    out = tx.apply(inp)
    mu = np.concatenate([np.asarray(out["tpit"]), np.asarray(out["tcore"])])
    if not jac:
        return mu, None
    J = tx.jacobian(inp, ["theta", "phi"], ["tpit", "tcore"])
    rows = [
        np.hstack([np.asarray(J[k]["theta"]), np.asarray(J[k]["phi"])])
        for k in ("tpit", "tcore")
    ]
    return mu, np.vstack(rows)


# --- unconstrained parameterization and priors -------------------------------


def _is_beta(d) -> bool:
    return isinstance(d, dist.Beta)


def to_z(x):
    """Constrained → unconstrained (log for LogNormal sites, logit for Beta sites)."""
    x = np.asarray(x, dtype=float)
    return np.array(
        [
            np.log(v / (1 - v)) if _is_beta(d) else np.log(v)
            for v, (_, d) in zip(x, SITES, strict=True)
        ]
    )


def from_z(z):
    """Unconstrained → constrained."""
    z = np.asarray(z, dtype=float)
    return np.array(
        [
            1 / (1 + np.exp(-v)) if _is_beta(d) else np.exp(v)
            for v, (_, d) in zip(z, SITES, strict=True)
        ]
    )


def dx_dz(z):
    """Diagonal of the transform Jacobian dx/dz."""
    x = from_z(z)
    return np.array(
        [v * (1 - v) if _is_beta(d) else v for v, (_, d) in zip(x, SITES, strict=True)]
    )


def prior_terms(z):
    """−log p(z), its gradient and its (diagonal) Hessian in the unconstrained space.

    LogNormal sites are Gaussian in z; Beta(a, b) sites carry the logit
    Jacobian, so −log p(z) = −a·log σ(z) − b·log(1 − σ(z)).
    """
    z = np.asarray(z, dtype=float)
    f, g, h = 0.0, np.zeros_like(z), np.zeros_like(z)
    for i, (v, (_, d)) in enumerate(zip(z, SITES, strict=True)):
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
    mu, J = model_mean(from_z(z), log, jac=jac)
    w = wh.residual(mu)
    fp, gp, hp = prior_terms(z)
    F = 0.5 * float(w @ w) + fp
    if not jac:
        return F, None, None, w
    Jw = wh.jacobian(J) * dx_dz(z)[None, :]
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

    The 14 physics coordinates get the Gauss-Newton covariance
    (Jᵀ Σ⁻¹ J + prior curvature)⁻¹ in the unconstrained space — ridge widths
    included, no eigenvalue floor; the 4 noise coordinates get the
    conditional block of the finite-difference Laplace Hessian when
    ``hess_npz`` is given, else the prior's scale. Assembled in numpyro's
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


def laplace_inverse_mass(hess_npz, floor_rel: float = 1e-4) -> np.ndarray:
    """Dense inverse mass matrix from a finite-difference Laplace Hessian.

    The npz (from run_laplace.py) holds the 18×18 Hessian of the potential in
    numpyro's sorted-site order; eigenvalues below ``floor_rel`` of the
    largest are floored so flat ridges do not blow the metric up.
    """
    c = np.load(hess_npz)
    H = 0.5 * (c["H"] + c["H"].T)
    lam, V = np.linalg.eigh(H)
    lam_reg = np.maximum(np.abs(lam), floor_rel * lam.max())
    return (V / lam_reg) @ V.T
