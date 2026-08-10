"""Phase B stage 2: full NUTS posterior, initialized at the MAP.

Run from app/ (intended as an overnight background run):
    uv run python -u ../scripts/run_nuts.py [--warmup N] [--samples N]

Saves data/calibration/nuts_samples.npz + diagnostics, then renders the
corner plot and posterior-predictive band (figures/).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Must be set BEFORE the tesseract api modules import: failed solves at
# pathological NUTS proposals become rejections, not chain crashes.
os.environ["BRISKET_SOFT_FAIL"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app.clients import meat_client, pit_client
from brisket_app import calibrate, synth
from brisket_app.coupled import CookGrids
from brisket_app.plots import corner_plot, posterior_predictive_plot

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=6)
    args = ap.parse_args()

    data = synth.load("cook01")
    meat_tx, pit_tx = meat_client(), pit_client()

    map_file = REPO / "data" / "calibration" / "map_result.json"
    init = {k: v for k, v in json.loads(map_file.read_text()).items()
            if not k.startswith("_")}
    print(f"init at MAP from {map_file.name}")

    t0 = time.perf_counter()
    mcmc = calibrate.run_nuts(data, meat_tx, pit_tx,
                              num_warmup=args.warmup, num_samples=args.samples,
                              max_tree_depth=args.max_tree_depth,
                              init_params=init)
    wall_h = (time.perf_counter() - t0) / 3600
    print(f"NUTS done in {wall_h:.2f} h")
    mcmc.print_summary(exclude_deterministic=True)

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    out = REPO / "data" / "calibration"
    np.savez(out / "nuts_samples.npz", **samples)

    truths = {n: t for n, t in zip(
        [n for n, _ in calibrate.THETA_SITES], data["theta_true"])}
    truths.update({n: t for n, t in zip(
        [n for n, _ in calibrate.PHI_SITES], data["phi_true"])})

    grids = CookGrids(t_end=float(data["t"][-1]),
                      dt_exchange=float(data["t"][1] - data["t"][0]),
                      n_u=len(data["u_knots"]))
    corner_plot(samples, REPO / "figures" / "corner.png", truths=truths)
    posterior_predictive_plot(samples, data, meat_tx, pit_tx, grids,
                              REPO / "figures" / "posterior_predictive.png")
    print("wrote figures/corner.png, figures/posterior_predictive.png")


if __name__ == "__main__":
    main()
