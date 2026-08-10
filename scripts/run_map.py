"""Phase B stage 1: synthetic data + gradient timing + MAP recovery.

Run from app/:  uv run python ../scripts/run_map.py [--steps N] [--skip-timing]

Stages (each prints before the next starts, so partial runs are informative):
  1. Generate + save the synthetic cook log (data/synthetic/cook01.npz)
  2. Time one log-density gradient  -> the NUTS go/no-go checkpoint number
  3. Run MAP (SVI AutoDelta), print the recovery table (Milestone B1)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app.clients import meat_client, pit_client
from brisket_app import synth, calibrate

RESULTS = Path(__file__).resolve().parents[1] / "data" / "calibration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--regen-data", action="store_true")
    args = ap.parse_args()

    meat_tx, pit_tx = meat_client(), pit_client()

    data_file = synth.DATA_DIR / "cook01.npz"
    if args.regen_data or not data_file.exists():
        print("== generating synthetic cook log ==")
        t0 = time.perf_counter()
        data = synth.generate(meat_tx, pit_tx)
        path = synth.save(data)
        print(f"wrote {path} ({time.perf_counter() - t0:.1f} s)")
    data = synth.load("cook01")
    print(f"data: {len(data['t'])} obs, core range "
          f"{data['obs_core'].min() - 273.15:.1f}-{data['obs_core'].max() - 273.15:.1f} C")

    if not args.skip_timing:
        print("== timing one log-density gradient (post-warmup) ==")
        s = calibrate.time_gradient(data, meat_tx, pit_tx)
        est = args.steps * s / 60
        print(f"s/gradient = {s:.2f}  ->  MAP({args.steps} steps) ~ {est:.0f} min")
        print(f"NUTS budget check: 600 draws x 2^6 leapfrog ~ {600 * 64 * s / 3600:.1f} h worst case")

    print(f"== MAP: {args.steps} steps, lr={args.lr} ==")
    map_params, losses = calibrate.run_map(data, meat_tx, pit_tx,
                                           num_steps=args.steps, lr=args.lr)
    print(calibrate.recovery_table(map_params, data))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {k: float(v) for k, v in map_params.items()}
    out["_final_loss"] = float(losses[-1])
    (RESULTS / "map_result.json").write_text(json.dumps(out, indent=2))
    np.save(RESULTS / "map_losses.npy", losses)
    print(f"wrote {RESULTS / 'map_result.json'}")


if __name__ == "__main__":
    main()
