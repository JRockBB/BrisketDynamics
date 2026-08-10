"""Phase B stage 2: full NUTS posterior, initialized at the MAP,
with chunked checkpointing so a ~day-long run survives interruption.

Run from app/ (intended as a detached long run):
    uv run python -u ../scripts/run_nuts.py [--warmup N] [--samples N] [--chunk N]

Checkpoints in data/calibration/nuts_ckpt/:
    state.pkl     sampler state after warmup / after each chunk (resume point)
    meta.json     run configuration + completed-chunk count
    chunk_NNN.npz samples of each completed chunk
Re-running the script resumes automatically: completed warmup is never
repeated, and at most one chunk of samples is lost. On completion the chunks
are concatenated to data/calibration/nuts_samples.npz and the corner +
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
from brisket_app import calibrate, synth
from brisket_app.coupled import CookGrids
from brisket_app.plots import corner_plot, posterior_predictive_plot

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "data" / "calibration" / "nuts_ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--max-tree-depth", type=int, default=6)
    args = ap.parse_args()

    from numpyro.infer import MCMC, NUTS
    import numpyro

    data = synth.load("cook01")
    meat_tx, pit_tx = meat_client(), pit_client()
    m = calibrate._bound_model(data, meat_tx, pit_tx)

    map_file = REPO / "data" / "calibration" / "map_result.json"
    init = {k: v for k, v in json.loads(map_file.read_text()).items()
            if not k.startswith("_")}

    kernel = NUTS(m, max_tree_depth=args.max_tree_depth,
                  init_strategy=numpyro.infer.init_to_value(values=init))
    n_chunks = (args.samples + args.chunk - 1) // args.chunk
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.chunk,
                progress_bar=True)

    CKPT.mkdir(parents=True, exist_ok=True)
    state_file = CKPT / "state.pkl"
    meta_file = CKPT / "meta.json"

    done_chunks = 0
    if state_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        if meta.get("config") == vars(args):
            with open(state_file, "rb") as f:
                mcmc.post_warmup_state = pickle.load(f)
            done_chunks = meta["done_chunks"]
            print(f"RESUMING: warmup done, {done_chunks}/{n_chunks} chunks complete",
                  flush=True)
        else:
            print("checkpoint config mismatch — starting fresh", flush=True)
            state_file.unlink()

    t0 = time.perf_counter()
    if not state_file.exists():
        print(f"warmup: {args.warmup} iterations (init at MAP)", flush=True)
        mcmc.warmup(jax.random.PRNGKey(0))
        with open(state_file, "wb") as f:
            pickle.dump(jax.device_get(mcmc.post_warmup_state), f)
        meta_file.write_text(json.dumps({"config": vars(args), "done_chunks": 0}))
        print(f"warmup done in {(time.perf_counter() - t0) / 3600:.2f} h; "
              "state checkpointed", flush=True)

    for c in range(done_chunks, n_chunks):
        tc = time.perf_counter()
        mcmc.run(mcmc.post_warmup_state.rng_key)
        np.savez(CKPT / f"chunk_{c:03d}.npz",
                 **{k: np.asarray(v) for k, v in mcmc.get_samples().items()})
        mcmc.post_warmup_state = mcmc.last_state
        with open(state_file, "wb") as f:
            pickle.dump(jax.device_get(mcmc.last_state), f)
        meta_file.write_text(json.dumps({"config": vars(args), "done_chunks": c + 1}))
        print(f"chunk {c + 1}/{n_chunks} done "
              f"({(time.perf_counter() - tc) / 60:.1f} min)", flush=True)

    # ---- assemble + figures ----
    chunks = [np.load(CKPT / f"chunk_{c:03d}.npz") for c in range(n_chunks)]
    samples = {k: np.concatenate([np.asarray(ch[k]) for ch in chunks])
               for k in chunks[0].files}
    out = REPO / "data" / "calibration"
    np.savez(out / "nuts_samples.npz", **samples)
    print(f"NUTS done in {(time.perf_counter() - t0) / 3600:.2f} h "
          f"({len(samples[next(iter(samples))])} samples)", flush=True)

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
    print("wrote figures/corner.png, figures/posterior_predictive.png", flush=True)


if __name__ == "__main__":
    main()
