"""Evening-scale ADVI on the coupled model. Does not replace NUTS.

Full-rank (default) / low-rank / mean-field SVI on the same NumPyro model
`run_nuts_mono.py` samples. Checkpoints the SVI state so a run can be paused.

Run from app/:
    uv run python -u ../scripts/run_vi.py --real [--steps N]

Writes:
    data/calibration/vi_ckpt[_real]/   state.pkl, meta.json
    data/calibration/vi_samples[_real].npz
    figures/corner_vi[_real].png
    figures/posterior_predictive_vi[_real].png
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ["BRISKET_SOFT_FAIL"] = "1"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax
import numpyro
from brisket_app import calibrate, cooklog, synth
from brisket_app.clients import monolith_client
from brisket_app.coupled import CookGrids
from brisket_app.plots import corner_plot, posterior_predictive_plot
from numpyro.infer import SVI, Trace_ELBO

REPO = Path(__file__).resolve().parents[1]
CKPT_EVERY = 5


def main():
    """Fit (or resume) ADVI, draw from the guide, render corner + predictive."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument(
        "--guide",
        choices=("full", "lowrank", "meanfield"),
        default="full",
        help="variational family (full-rank Gaussian can represent ridges)",
    )
    ap.add_argument("--rank", type=int, default=8, help="lowrank factor rank")
    ap.add_argument(
        "--init-scale",
        type=float,
        default=0.1,
        help="initial guide scale in unconstrained space",
    )
    ap.add_argument("--draws", type=int, default=400, help="posterior draws to save")
    ap.add_argument(
        "--real",
        action="store_true",
        help="calibrate against the real cook log instead of synthetic",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--init-json",
        default=None,
        help="MAP json to initialize the guide loc (default: map[_real].json)",
    )
    args = ap.parse_args()
    tag = "_real" if args.real else ""
    CKPT = REPO / "data" / "calibration" / f"vi_ckpt{tag}"

    tx = monolith_client()
    if args.real:
        log = cooklog.load(dt_min=10.0)
        model = calibrate.bound_model_real(log, tx)
        map_file = REPO / "data" / "calibration" / "map_real.json"
    else:
        data = synth.load("cook01")
        model = calibrate._bound_model(data, tx)
        map_file = REPO / "data" / "calibration" / "map_result.json"
    if args.init_json:
        map_file = Path(args.init_json)
    init = {}
    if map_file.exists():
        init = {
            k: float(v)
            for k, v in json.loads(map_file.read_text()).items()
            if not k.startswith("_")
        }

    guide = calibrate.make_vi_guide(
        model,
        kind=args.guide,
        init_values=init or None,
        rank=args.rank,
        init_scale=args.init_scale,
    )
    svi = SVI(model, guide, numpyro.optim.Adam(args.lr), Trace_ELBO())

    CKPT.mkdir(parents=True, exist_ok=True)
    state_file = CKPT / "state.pkl"
    meta_file = CKPT / "meta.json"

    rng = jax.random.PRNGKey(args.seed)
    step0 = 0
    # a change in the latent set or in the MAP the run is built on (a new
    # calibration of new physics) starts fresh
    sites = [n for n, _ in calibrate.THETA_SITES]
    init_stamp = hashlib.sha1(map_file.read_bytes()).hexdigest() if map_file.exists() else ""
    if state_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        old = dict(meta.get("config") or {})
        new = vars(args)
        same = (
            {k: v for k, v in old.items() if k != "steps"}
            == {k: v for k, v in new.items() if k != "steps"}
            and old.get("steps", -1) <= new["steps"]
            and meta.get("sites") == sites
            and meta.get("init") == init_stamp
        )
        if same:
            svi.init(rng)  # builds svi.constrain_fn; the state itself is the checkpoint
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            step0 = int(meta.get("step", 0))
            print(f"RESUMING VI at step {step0}/{args.steps}", flush=True)
        else:
            print("VI checkpoint config mismatch — starting fresh", flush=True)
            state_file.unlink()
            state = svi.init(rng)
    else:
        state = svi.init(rng)

    def save(st, i: int, done: bool = False):
        with open(state_file, "wb") as f:
            pickle.dump(jax.device_get(st), f)
        meta_file.write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "sites": sites,
                    "init": init_stamp,
                    "step": i,
                    "done": done,
                }
            )
        )

    t0 = time.perf_counter()
    for i in range(step0, args.steps):
        state, loss = svi.update(state)
        n = i + 1
        if n % CKPT_EVERY == 0 or n == args.steps:
            save(state, n, done=(n == args.steps))
        if n % 5 == 0 or n == args.steps:
            print(
                f"vi {n}/{args.steps}  loss={float(loss):.1f}  "
                f"({(time.perf_counter() - t0) / 60:.1f} min elapsed)",
                flush=True,
            )

    save(state, args.steps, done=True)
    params = svi.get_params(state)
    stacked = calibrate.draw_vi_posterior(guide, params, args.draws, seed=args.seed + 1)
    out = REPO / "data" / "calibration" / f"vi_samples{tag}.npz"
    np.savez(out, **stacked)
    print(f"wrote {out.name} ({args.draws} draws)", flush=True)

    if args.real:
        grids = CookGrids(
            t_end=float(log["t"][-1]),
            dt_exchange=float(log["t"][1] - log["t"][0]),
            n_u=len(log["t"]),
        )
        pp_data = {
            "u_knots": log["u"],
            "obs_pit": log["obs_pit"],
            "obs_core": log["obs_core"],
        }
        ics = {
            "tpit0": log["meta"]["tpit0_K"],
            "t_amb": 300.0,
            "t0_meat": log["meta"]["tcore0_K"],
        }
        names = (
            [n for n, _ in calibrate.THETA_SITES]
            + [n for n, _ in calibrate.PHI_SITES]
            + [n for n, _ in calibrate.PHI_SITES_IGNITION]
            + [n for n, _ in calibrate.RHO_SITES]
        )
        corner_plot(stacked, REPO / "figures" / "corner_vi_real.png", names=names)
        posterior_predictive_plot(
            stacked,
            pp_data,
            tx,
            grids,
            REPO / "figures" / "posterior_predictive_vi_real.png",
            ics=ics,
            masks={"pit": log["mask_pit"], "core": log["mask_core"]},
        )
    else:
        truths = {
            n: t
            for n, t in zip(
                [n for n, _ in calibrate.THETA_SITES], data["theta_true"], strict=True
            )
        }
        truths.update(
            {
                n: t
                for n, t in zip(
                    [n for n, _ in calibrate.PHI_SITES], data["phi_true"], strict=False
                )
            }
        )
        grids = CookGrids(
            t_end=float(data["t"][-1]),
            dt_exchange=float(data["t"][1] - data["t"][0]),
            n_u=len(data["u_knots"]),
        )
        corner_plot(stacked, REPO / "figures" / "corner_vi.png", truths=truths)
        posterior_predictive_plot(
            stacked,
            data,
            tx,
            grids,
            REPO / "figures" / "posterior_predictive_vi.png",
        )
    print(f"wrote figures/corner_vi{tag}.png, figures/posterior_predictive_vi{tag}.png")


if __name__ == "__main__":
    main()
