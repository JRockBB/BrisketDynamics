"""Merge the monolith NUTS chains, print diagnostics, compare posteriors.

Run from app/:  uv run python -u ../scripts/merge_nuts_mono.py
Reads data/calibration/nuts_mono_real/chain*.npz, writes
data/calibration/nuts_mono_samples_real.npz (site name -> draws, the
format robust control and the posterior-predictive plots read) and prints
split-r-hat / ESS per site plus the sd(z) of NUTS against the Fisher
(gn_map_real.json), ADVI (vi_samples_real.npz) and Laplace
(laplace_samples_real.npz) posteriors on the same unconstrained coordinates.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app.monolith import ALL_NAMES_SORTED, NAMES, NOISE_NAMES, to_z
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

REPO = Path(__file__).resolve().parents[1]


def sd_z(samples: dict) -> dict:
    """Posterior sd in the unconstrained coordinates, per physics site."""
    Z = np.stack(
        [to_z([samples[n][i] for n in NAMES]) for i in range(len(samples[NAMES[0]]))]
    )
    return dict(zip(NAMES, np.std(Z, axis=0), strict=True))


def main():
    """Merge chains, write the sample file, print diagnostics and the width table."""
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
        ("ADVI", "vi_samples_real.npz"),
        ("Laplace-GN", "laplace_samples_real.npz"),
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


if __name__ == "__main__":
    main()
