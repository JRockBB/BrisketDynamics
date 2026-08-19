"""Phase B stage 2: full NUTS posterior, initialized at the MAP.

EVERY step — warmup included — is checkpointed, via NumPyro's kernel-level
API (`kernel.init` / `kernel.sample`): the HMCState namedtuple carries the
complete sampler + adaptation state and pickles cleanly, so an interruption
at any point (including hour 8 of warmup) resumes from the last checkpoint.
The MCMC convenience wrapper cannot do this; this runner deliberately avoids
it.

Run from app/ (intended as a detached long run):
    uv run python -u ../scripts/run_nuts.py [--warmup N] [--samples N]

Checkpoints in data/calibration/nuts_ckpt/:
    state.pkl      HMCState at the last checkpointed step (resume point)
    meta.json      run configuration
    samples.npz    constrained samples collected so far (rewritten each save)
Re-running the same command resumes automatically. On completion, samples are
copied to data/calibration/nuts_samples.npz and the corner +
posterior-predictive figures are rendered.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Must be set BEFORE the tesseract api modules import: failed solves at
# pathological NUTS proposals become rejections, not chain crashes.
os.environ["BRISKET_SOFT_FAIL"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax

from brisket_app.clients import meat_client, pit_client
from brisket_app import calibrate, cooklog, synth
from brisket_app.coupled import CookGrids
from brisket_app.plots import corner_plot, posterior_predictive_plot

REPO = Path(__file__).resolve().parents[1]
CKPT_EVERY = 5  # steps between state checkpoints (~10 min of compute at risk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=6)
    ap.add_argument("--real", action="store_true",
                    help="calibrate against the real cook log (data/cook_log/) "
                         "instead of the synthetic dataset")
    ap.add_argument("--chain", default="",
                    help="chain tag for multi-chain runs: suffixes the "
                         "checkpoint dir and samples file, skips figures")
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed")
    ap.add_argument("--init-json", default=None,
                    help="override init-value file (default: the MAP json)")
    args = ap.parse_args()
    total = args.warmup + args.samples
    chain = f"_{args.chain}" if args.chain else ""
    tag = ("_real" if args.real else "") + chain
    CKPT = REPO / "data" / "calibration" / f"nuts_ckpt{tag}"

    import numpyro
    from numpyro.infer import NUTS

    meat_tx, pit_tx = meat_client(), pit_client()
    if args.real:
        log = cooklog.load(dt_min=10.0)
        model = calibrate.bound_model_real(log, meat_tx, pit_tx)
        map_file = REPO / "data" / "calibration" / "map_real.json"
    else:
        data = synth.load("cook01")
        model = calibrate._bound_model(data, meat_tx, pit_tx)
        map_file = REPO / "data" / "calibration" / "map_result.json"
    if args.init_json:
        map_file = Path(args.init_json)
    init = {k: v for k, v in json.loads(map_file.read_text()).items()
            if not k.startswith("_")}

    kernel = NUTS(model, max_tree_depth=args.max_tree_depth,
                  init_strategy=numpyro.infer.init_to_value(values=init))
    # kernel.init sets up the adaptation schedule and potential fn; it must
    # run on every start INCLUDING resumes (the pickled state then replaces
    # the fresh one).
    state = kernel.init(jax.random.PRNGKey(args.seed), args.warmup,
                        init_params=None, model_args=(), model_kwargs={})
    postprocess = kernel.postprocess_fn((), {})

    CKPT.mkdir(parents=True, exist_ok=True)
    state_file = CKPT / "state.pkl"
    meta_file = CKPT / "meta.json"
    samples_file = CKPT / "samples.npz"

    samples: list[dict] = []
    if state_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("config") == vars(args):
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            if samples_file.exists():
                with np.load(samples_file) as f:
                    keys = f.files
                    n = len(f[keys[0]])
                    samples = [{k: f[k][i] for k in keys} for i in range(n)]
            print(f"RESUMING at step {int(state.i)}/{total} "
                  f"({len(samples)} samples already collected)", flush=True)
        else:
            print("checkpoint config mismatch — starting fresh", flush=True)
            state_file.unlink()

    step = jax.jit(lambda s: kernel.sample(s, (), {}))

    def save(st, done: bool = False):
        with open(state_file, "wb") as f:
            pickle.dump(jax.device_get(st), f)
        meta_file.write_text(json.dumps({"config": vars(args), "done": done}))
        if samples:
            np.savez(samples_file,
                     **{k: np.stack([s[k] for s in samples])
                        for k in samples[0]})

    t0 = time.perf_counter()
    i = int(state.i)
    while i < total:
        state = step(state)
        jax.block_until_ready(state.z)
        i = int(state.i)
        phase = "warmup" if i <= args.warmup else "sample"
        if i > args.warmup:
            samples.append({k: np.asarray(v)
                            for k, v in postprocess(state.z).items()})
        if i % CKPT_EVERY == 0 or i == total:
            save(state)
        if i % 5 == 0 or i == total:
            print(f"{phase} {i}/{total}  "
                  f"({(time.perf_counter() - t0) / 60:.1f} min elapsed, "
                  f"{len(samples)} samples)", flush=True)

    save(state, done=True)
    out = REPO / "data" / "calibration"
    stacked = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
    np.savez(out / f"nuts_samples{tag}.npz", **stacked)
    print(f"NUTS done in {(time.perf_counter() - t0) / 3600:.2f} h "
          f"({len(samples)} samples)", flush=True)

    if args.chain:
        print(f"chain {args.chain}: samples saved; figures skipped", flush=True)
        return
    if args.real:
        grids = CookGrids(t_end=float(log["t"][-1]),
                          dt_exchange=float(log["t"][1] - log["t"][0]),
                          n_u=len(log["t"]))
        pp_data = {"u_knots": log["u"], "obs_pit": log["obs_pit"],
                   "obs_core": log["obs_core"]}
        ics = {"tpit0": log["meta"]["tpit0_K"], "t_amb": 300.0,
               "t0_meat": log["meta"]["tcore0_K"]}
        real_names = ([n for n, _ in calibrate.THETA_SITES]
              + [n for n, _ in calibrate.PHI_SITES]
              + [n for n, _ in calibrate.PHI_SITES_IGNITION])
        corner_plot(stacked, REPO / "figures" / "corner_real.png",
                    names=real_names)  # no truths: real data
        posterior_predictive_plot(
            stacked, pp_data, meat_tx, pit_tx, grids,
            REPO / "figures" / "posterior_predictive_real.png",
            ics=ics, masks={"pit": log["mask_pit"], "core": log["mask_core"]})
    else:
        truths = {n: t for n, t in zip(
            [n for n, _ in calibrate.THETA_SITES], data["theta_true"])}
        truths.update({n: t for n, t in zip(
            [n for n, _ in calibrate.PHI_SITES], data["phi_true"])})
        grids = CookGrids(t_end=float(data["t"][-1]),
                          dt_exchange=float(data["t"][1] - data["t"][0]),
                          n_u=len(data["u_knots"]))
        corner_plot(stacked, REPO / "figures" / "corner.png", truths=truths)
        posterior_predictive_plot(stacked, data, meat_tx, pit_tx, grids,
                                  REPO / "figures" / "posterior_predictive.png")
    print(f"wrote figures/corner{tag}.png, figures/posterior_predictive{tag}.png",
          flush=True)


if __name__ == "__main__":
    main()
