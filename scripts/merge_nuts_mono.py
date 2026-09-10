"""Merge the monolith NUTS chains, print diagnostics, compare posteriors.

Run from app/:  uv run python -u ../scripts/merge_nuts_mono.py [--no-figures]
Reads data/calibration/nuts_mono_real/chain*.npz, writes
data/calibration/nuts_mono_samples_real.npz (site name -> draws, the
format robust control and the posterior-predictive plots read) and prints
split-r-hat / ESS per site plus the sd(z) of NUTS against the Fisher
(gn_map_real.json), the Laplace (laplace_samples_real.npz) and the TMB
nested-ignition posteriors on the same unconstrained coordinates.

Writes figures/calibration/corner_nuts_real.png, figures/calibration/posterior_predictive_nuts_real.png
(the reference posterior gets the same two figures as the approximations) and
figures/calibration/posterior_widths.png, the width comparison the table above prints.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app.monolith import ALL_NAMES_SORTED, NAMES, NOISE_NAMES, to_z
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

REPO = Path(__file__).resolve().parents[1]


def _figures(merged: dict, cols: dict):
    """Corner + posterior predictive for the NUTS draws, and the width panel."""
    from brisket_app import calibrate, cooklog
    from brisket_app.clients import monolith_client
    from brisket_app.coupled import CookGrids
    from brisket_app.plots import (
        corner_plot,
        posterior_predictive_plot,
        posterior_width_plot,
    )

    posterior_width_plot(cols, REPO / "figures" / "calibration" / "posterior_widths.png")
    print("wrote figures/calibration/posterior_widths.png")

    corner_names = (
        [k for k, _ in calibrate.THETA_SITES]
        + [k for k, _ in calibrate.PHI_SITES]
        + [k for k, _ in calibrate.PHI_SITES_IGNITION]
        + [k for k, _ in calibrate.RHO_SITES]
    )
    corner_plot(merged, REPO / "figures" / "calibration" / "corner_nuts_real.png", names=corner_names)
    print("wrote figures/calibration/corner_nuts_real.png")

    log = cooklog.load(dt_min=10.0)
    tx = monolith_client()
    grids = CookGrids(
        t_end=float(log["t"][-1]),
        dt_exchange=float(log["t"][1] - log["t"][0]),
        n_u=len(log["t"]),
    )
    posterior_predictive_plot(
        merged,
        {"u_knots": log["u"], "obs_pit": log["obs_pit"], "obs_core": log["obs_core"]},
        tx,
        grids,
        REPO / "figures" / "calibration" / "posterior_predictive_nuts_real.png",
        ics={
            "tpit0": log["meta"]["tpit0_K"],
            "t_amb": 300.0,
            "t0_meat": log["meta"]["tcore0_K"],
        },
        masks={"pit": log["mask_pit"], "core": log["mask_core"]},
    )
    print("wrote figures/calibration/posterior_predictive_nuts_real.png")


def sd_z(samples: dict) -> dict:
    """Posterior sd in the unconstrained coordinates, per physics site."""
    Z = np.stack(
        [to_z([samples[n][i] for n in NAMES]) for i in range(len(samples[NAMES[0]]))]
    )
    return dict(zip(NAMES, np.std(Z, axis=0), strict=True))


def main():
    """Merge chains, write the sample file, print diagnostics and the width table."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="diagnostics only; skip the corner, predictive and width figures",
    )
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"
    files = sorted((cal / "nuts_mono_real").glob("chain*.npz"))
    if not files:
        raise SystemExit("no chains found under data/calibration/nuts_mono_real")
    chains = [dict(np.load(f)) for f in files]
    n = min(len(c[ALL_NAMES_SORTED[0]]) for c in chains)
    print(f"{len(chains)} chains, {n} samples each (truncated to the shortest)")
    stacked = {k: np.stack([c[k][:n] for c in chains]) for k in ALL_NAMES_SORTED}
    print(f"{'site':10s} {'r-hat':>7s} {'ESS':>7s} {'mean':>11s} {'sd':>10s}")
    for k in ALL_NAMES_SORTED:
        x = stacked[k]
        rhat = float(split_gelman_rubin(x)) if len(chains) > 1 else float("nan")
        ess = float(effective_sample_size(x))
        print(f"{k:10s} {rhat:7.3f} {ess:7.0f} {x.mean():11.4g} {x.std():10.3g}")
    div = sum(int(np.sum(c["_diverging"][:n])) for c in chains if "_diverging" in c)
    steps = np.mean([np.mean(c["_num_steps"][:n]) for c in chains if "_num_steps" in c])
    print(f"divergences {div}, mean leapfrog steps {steps:.1f}")

    merged = {k: np.concatenate([c[k][:n] for c in chains]) for k in ALL_NAMES_SORTED}
    out = cal / "nuts_mono_samples_real.npz"
    np.savez(out, **merged)
    print(f"wrote {out.name} ({len(merged[NAMES[0]])} draws)")

    cols = {"NUTS-mono": sd_z(merged)}
    gn = cal / "gn_map_real.json"
    if gn.exists():
        cols["Fisher"] = json.loads(gn.read_text())["sd_z_fisher"]
    for label, f in (
        ("Laplace-GN", "laplace_samples_real.npz"),
        ("TMB-ign", "tmb_ignition_samples_real.npz"),
        ("TMB-PSIS", "psis_tmb_samples_real.npz"),
    ):
        if (cal / f).exists():
            cols[label] = sd_z(dict(np.load(cal / f)))
    print("\nsd(z) per physics site:")
    print(f"{'site':10s}" + "".join(f"{c:>12s}" for c in cols))
    for nme in NAMES:
        print(f"{nme:10s}" + "".join(f"{cols[c][nme]:12.3f}" for c in cols))
    print(
        "noise sites (constrained):",
        {k: f"{merged[k].mean():.3f}+-{merged[k].std():.3f}" for k in NOISE_NAMES},
    )
    if not args.no_figures:
        _figures(merged, cols)


if __name__ == "__main__":
    main()
