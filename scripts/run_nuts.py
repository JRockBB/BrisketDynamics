"""Checkpointed NUTS. Default is two chains; `--n-chains` sets the count.

Every step — warmup included — is pickled via NumPyro's kernel API
(`kernel.init` / `kernel.sample`). Re-running the same command resumes.

From app/:
    uv run python -u ../scripts/run_nuts.py --real
    uv run python -u ../scripts/run_nuts.py --real --n-chains 1

Checkpoints in data/calibration/nuts_ckpt[_real]/<chain>/ :
    state.pkl, meta.json, samples.npz, diagnostics.npz
On completion, samples are concatenated to nuts_samples[_real].npz and
the corner + posterior-predictive figures are rendered.
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Must be set BEFORE the tesseract api modules import: failed solves at
# pathological NUTS proposals become rejections, not chain crashes.
os.environ["BRISKET_SOFT_FAIL"] = "1"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax
from brisket_app import calibrate, cooklog, synth
from brisket_app.clients import monolith_client
from brisket_app.coupled import CookGrids
from brisket_app.plots import corner_plot, posterior_predictive_plot

REPO = Path(__file__).resolve().parents[1]
CKPT_EVERY = 5


def _cfg(args):
    return {
        "warmup": args.warmup,
        "samples": args.samples,
        "max_tree_depth": args.max_tree_depth,
        "real": args.real,
        "n_chains": args.n_chains,
        "seed": args.seed,
        "init_json": args.init_json,
    }


def _resume_ok(old, new):
    """Return True if the checkpoint can be resumed (samples may increase)."""
    skip = {"samples"}
    return {k: v for k, v in old.items() if k not in skip} == {
        k: v for k, v in new.items() if k not in skip
    } and old.get("samples", -1) <= new["samples"]


def _run_one_chain(kernel, postprocess, step, chain_dir, cfg, chain_id, total):
    """Advance one chain from init or checkpoint. Returns sample list."""
    chain_dir.mkdir(parents=True, exist_ok=True)
    state_file = chain_dir / "state.pkl"
    meta_file = chain_dir / "meta.json"
    samples_file = chain_dir / "samples.npz"
    diags_file = chain_dir / "diagnostics.npz"
    warmup = cfg["warmup"]

    state = kernel.init(
        jax.random.PRNGKey(cfg["seed"] + chain_id),
        warmup,
        init_params=None,
        model_args=(),
        model_kwargs={},
    )
    samples: list[dict] = []
    diags: list[dict] = []
    if state_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        old_cfg = dict(meta.get("config") or {})
        if _resume_ok(old_cfg, cfg) and not meta.get("done"):
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            if samples_file.exists():
                with np.load(samples_file) as f:
                    keys = f.files
                    n = len(f[keys[0]])
                    samples = [{k: f[k][i] for k in keys} for i in range(n)]
            if diags_file.exists():
                with np.load(diags_file) as f:
                    dkeys = f.files
                    nd = len(f[dkeys[0]])
                    diags = [{k: f[k][i] for k in dkeys} for i in range(nd)]
            print(
                f"chain {chain_id} RESUMING at step {int(state.i)}/{total} "
                f"({len(samples)} samples)",
                flush=True,
            )
        elif meta.get("done") and _resume_ok(old_cfg, cfg):
            if samples_file.exists():
                with np.load(samples_file) as f:
                    keys = f.files
                    n = len(f[keys[0]])
                    samples = [{k: f[k][i] for k in keys} for i in range(n)]
            print(
                f"chain {chain_id} already done ({len(samples)} samples)",
                flush=True,
            )
            return samples
        else:
            print(f"chain {chain_id} checkpoint mismatch — starting fresh", flush=True)

    def save(st, done: bool = False):
        with open(state_file, "wb") as f:
            pickle.dump(jax.device_get(st), f)
        meta_file.write_text(
            json.dumps({"config": cfg, "done": done, "chain": chain_id})
        )
        if samples:
            np.savez(
                samples_file,
                **{k: np.stack([s[k] for s in samples]) for k in samples[0]},
            )
        if diags:
            np.savez(
                diags_file,
                **{k: np.stack([d[k] for d in diags]) for k in diags[0]},
            )

    t0 = time.perf_counter()
    i = int(state.i)
    while i < total:
        state = step(state)
        jax.block_until_ready(state.z)
        i = int(state.i)
        phase = "warmup" if i <= warmup else "sample"
        if i > warmup:
            samples.append({k: np.asarray(v) for k, v in postprocess(state.z).items()})
            diags.append(
                {
                    "diverging": np.asarray(state.diverging, dtype=np.int8),
                    "accept_prob": np.asarray(state.accept_prob, dtype=np.float64),
                    "potential_energy": np.asarray(
                        state.potential_energy, dtype=np.float64
                    ),
                }
            )
        if i % CKPT_EVERY == 0 or i == total:
            save(state)
        if i % 5 == 0 or i == total:
            print(
                f"chain {chain_id} {phase} {i}/{total}  "
                f"({(time.perf_counter() - t0) / 60:.1f} min, "
                f"{len(samples)} samples)",
                flush=True,
            )

    save(state, done=True)
    if diags:
        n_div = int(np.sum([d["diverging"] for d in diags]))
        print(
            f"chain {chain_id} done: {len(samples)} samples, "
            f"{n_div}/{len(diags)} divergences, "
            f"mean accept={np.mean([float(d['accept_prob']) for d in diags]):.3f}",
            flush=True,
        )
    else:
        print(f"chain {chain_id} done: {len(samples)} samples", flush=True)
    return samples


def _setup_kernel(cfg):
    """Build the NUTS kernel and a jitted step for this process."""
    import numpyro
    from numpyro.infer import NUTS

    tx = monolith_client()
    if cfg["real"]:
        log = cooklog.load(dt_min=10.0)
        model = calibrate.bound_model_real(log, tx)
        map_file = REPO / "data" / "calibration" / "map_real.json"
        data = None
    else:
        log = None
        data = synth.load("cook01")
        model = calibrate._bound_model(data, tx)
        map_file = REPO / "data" / "calibration" / "map_result.json"
    if cfg["init_json"]:
        map_file = Path(cfg["init_json"])
    init = {
        k: v
        for k, v in json.loads(map_file.read_text()).items()
        if not k.startswith("_")
    }
    kernel = NUTS(
        model,
        max_tree_depth=cfg["max_tree_depth"],
        init_strategy=numpyro.infer.init_to_value(values=init),
    )
    kernel.init(
        jax.random.PRNGKey(cfg["seed"]),
        cfg["warmup"],
        init_params=None,
        model_args=(),
        model_kwargs={},
    )
    return {
        "kernel": kernel,
        "postprocess": kernel.postprocess_fn((), {}),
        "step": jax.jit(lambda s: kernel.sample(s, (), {})),
        "tx": tx,
        "log": log,
        "data": data,
    }


def _chain_worker(chain_id, cfg, ckpt, total):
    """Process entry: one chain, own Tesseract clients, own checkpoint dir."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("BRISKET_SOFT_FAIL", "1")
    print(f"chain {chain_id} process start", flush=True)
    s = _setup_kernel(cfg)
    _run_one_chain(
        s["kernel"],
        s["postprocess"],
        s["step"],
        Path(ckpt) / str(chain_id),
        cfg,
        chain_id,
        total,
    )


def _stack_chain_samples(ckpt, n_chains):
    """Concatenate per-chain samples.npz files along the draw axis."""
    chunks = []
    for c in range(n_chains):
        path = Path(ckpt) / str(c) / "samples.npz"
        if not path.exists():
            raise SystemExit(f"chain {c} wrote no samples at {path}")
        with np.load(path) as f:
            chunks.append({k: np.asarray(f[k]) for k in f.files})
    keys = chunks[0].keys()
    return {k: np.concatenate([ch[k] for ch in chunks], axis=0) for k in keys}


def main():
    """Run (or resume) NUTS for `--n-chains` chains, then render figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=6)
    ap.add_argument(
        "--real",
        action="store_true",
        help="calibrate against the real cook log (data/cook_log/) "
        "instead of the synthetic dataset",
    )
    ap.add_argument(
        "--n-chains",
        type=int,
        default=2,
        help="number of NUTS chains (default 2); seeds are --seed, --seed+1, ...",
    )
    ap.add_argument("--seed", type=int, default=0, help="PRNG seed of chain 0")
    ap.add_argument(
        "--init-json",
        default=None,
        help="override init-value file (default: the MAP json)",
    )
    args = ap.parse_args()
    if args.n_chains < 1:
        raise SystemExit("--n-chains must be >= 1")
    total = args.warmup + args.samples
    cfg = _cfg(args)
    tag = "_real" if args.real else ""
    ckpt = REPO / "data" / "calibration" / f"nuts_ckpt{tag}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "meta.json").write_text(json.dumps({"config": cfg, "done": False}))
    print(
        f"NUTS {args.n_chains} chain(s) in parallel, {args.warmup}+{args.samples} "
        f"steps, seed {args.seed}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=_chain_worker,
            args=(c, cfg, str(ckpt), total),
            name=f"nuts-{c}",
        )
        for c in range(args.n_chains)
    ]
    for p in procs:
        p.start()
        print(f"started {p.name} pid={p.pid}", flush=True)
    failed = []
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((p.name, p.exitcode))
    if failed:
        raise SystemExit(f"chain worker(s) failed: {failed}")

    stacked = _stack_chain_samples(ckpt, args.n_chains)
    out = REPO / "data" / "calibration"
    np.savez(out / f"nuts_samples{tag}.npz", **stacked)
    (ckpt / "meta.json").write_text(json.dumps({"config": cfg, "done": True}))
    print(
        f"NUTS done ({args.n_chains} chains, {stacked[next(iter(stacked))].shape[0]} draws)",
        flush=True,
    )

    s = _setup_kernel(cfg)
    tx, log, data = s["tx"], s["log"], s["data"]

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
        real_names = (
            [n for n, _ in calibrate.THETA_SITES]
            + [n for n, _ in calibrate.PHI_SITES]
            + [n for n, _ in calibrate.PHI_SITES_IGNITION]
            + [n for n, _ in calibrate.RHO_SITES]
        )
        corner_plot(stacked, REPO / "figures" / "corner_real.png", names=real_names)
        posterior_predictive_plot(
            stacked,
            pp_data,
            tx,
            grids,
            REPO / "figures" / "posterior_predictive_real.png",
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
        corner_plot(stacked, REPO / "figures" / "corner.png", truths=truths)
        posterior_predictive_plot(
            stacked,
            data,
            tx,
            grids,
            REPO / "figures" / "posterior_predictive.png",
        )
    print(
        f"wrote figures/corner{tag}.png, figures/posterior_predictive{tag}.png",
        flush=True,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
