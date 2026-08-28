"""NUTS on the monolith: one chain per process, checkpointed every step.

The model is `calibrate.model_real` on the monolith Tesseract; every
leapfrog step costs one forward-mode Jacobian over the 15 physics latents
(~2.5 s on the real-cook grid). Initialised at the Gauss-Newton MAP, dense
mass matrix seeded from the Fisher covariance (noise block from
run_laplace.py's Hessian) and adapted during warmup.

Every step — warmup included — is advanced through NumPyro's kernel API
(`kernel.init` / `kernel.sample`) and the kernel state is pickled every
CKPT_EVERY steps, so a power loss costs at most CKPT_EVERY steps: re-running
the same command resumes (`--samples` may be raised on resume).

Run from app/:  uv run python -u ../scripts/run_nuts_mono.py --chain 0
Writes data/calibration/nuts_mono_real/chain{c}.npz (samples + `_num_steps`,
`_diverging`, `_accept_prob`, `_potential_energy`), chain{c}.json and
chain{c}_state.pkl; merge with scripts/merge_nuts_mono.py.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax
from brisket_app import calibrate, cooklog
from brisket_app.calibrate import phi_full
from brisket_app.clients import monolith_client
from brisket_app.monolith import (
    ALL_NAMES_SORTED,
    N_THETA,
    NAMES,
    NOISE_NAMES,
    fisher_inverse_mass,
)
from numpyro.infer import NUTS, init_to_value

REPO = Path(__file__).resolve().parents[1]
CKPT_EVERY = 5
EXTRA = ("num_steps", "diverging", "accept_prob", "potential_energy")


def _resume_ok(old: dict, new: dict) -> bool:
    """Return True when every setting but the sample target matches the checkpoint."""
    skip = {"samples"}
    return {k: v for k, v in old.items() if k not in skip} == {
        k: v for k, v in new.items() if k not in skip
    } and old.get("samples", -1) <= new["samples"]


def _load_lists(npz_path: Path) -> tuple[list[dict], list[dict]]:
    """Load samples and per-step extras back into per-step dicts."""
    samples: list[dict] = []
    extras: list[dict] = []
    if not npz_path.exists():
        return samples, extras
    with np.load(npz_path) as f:
        keys = [k for k in f.files if not k.startswith("_")]
        ekeys = [k for k in f.files if k.startswith("_")]
        n = len(f[keys[0]]) if keys else 0
        samples = [{k: f[k][i] for k in keys} for i in range(n)]
        extras = [{k[1:]: f[k][i] for k in ekeys} for i in range(n)]
    return samples, extras


def main():
    """Run (or resume) one NUTS chain on the monolith."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None, help="default: chain index")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--max-tree-depth", type=int, default=5)
    ap.add_argument("--init-physics", default="gn_map_real.json")
    ap.add_argument("--init-noise", default="map_real.json")
    ap.add_argument(
        "--mass",
        default="fisher",
        help="'fisher' (Gauss-Newton covariance at the init, noise block from "
        "laplace_hess_real.npz) or '' = identity",
    )
    args = ap.parse_args()
    seed = args.chain if args.seed is None else args.seed
    cal = REPO / "data" / "calibration"
    out_dir = cal / "nuts_mono_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "chain": args.chain,
        "seed": seed,
        "warmup": args.warmup,
        "samples": args.samples,
        "max_tree_depth": args.max_tree_depth,
        "mass": args.mass,
        "init_physics": args.init_physics,
        "init_noise": args.init_noise,
    }
    total = args.warmup + args.samples
    state_file = out_dir / f"chain{args.chain}_state.pkl"
    meta_path = out_dir / f"chain{args.chain}.json"
    npz_path = out_dir / f"chain{args.chain}.npz"

    gn = json.loads((cal / args.init_physics).read_text())
    noise = json.loads((cal / args.init_noise).read_text())
    init = {n: float(gn["x"][n]) for n in NAMES} if "x" in gn else {}
    if not init:  # a plain MAP json
        init = {n: float(gn[n]) for n in NAMES[:N_THETA]}
        init.update(dict(zip(NAMES[N_THETA:], phi_full(gn), strict=True)))
    init.update({n: float(noise[n]) for n in NOISE_NAMES})

    log = cooklog.load(dt_min=10.0)
    model = calibrate.bound_model_real(log, monolith_client())
    if args.mass == "fisher":
        hess = cal / "laplace_hess_real.npz"
        inv_mass = fisher_inverse_mass(
            np.array([init[n] for n in NAMES]),
            log,
            noise,
            hess if hess.exists() else None,
        )
    else:
        inv_mass = None
    kernel = NUTS(
        model,
        dense_mass=True,
        inverse_mass_matrix=inv_mass,
        init_strategy=init_to_value(values=init),
        max_tree_depth=args.max_tree_depth,
    )
    state = kernel.init(
        jax.random.PRNGKey(seed),
        args.warmup,
        init_params=None,
        model_args=(),
        model_kwargs={},
    )
    postprocess = kernel.postprocess_fn((), {})
    step = jax.jit(lambda s: kernel.sample(s, (), {}))

    samples: list[dict] = []
    extras: list[dict] = []
    wall_prev = 0.0
    if state_file.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        old = dict(meta.get("config") or {})
        extend = args.samples > int(meta.get("samples", 0))
        if _resume_ok(old, cfg) and (not meta.get("done") or extend):
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            samples, extras = _load_lists(npz_path)
            wall_prev = float(meta.get("wall_s", 0.0))
            print(
                f"chain {args.chain} RESUMING at step {int(state.i)}/{total} "
                f"({len(samples)} samples)",
                flush=True,
            )
        elif _resume_ok(old, cfg):
            print(f"chain {args.chain} already done ({meta.get('samples')} samples)")
            return
        else:
            print(
                f"chain {args.chain} checkpoint mismatch - starting fresh", flush=True
            )
    print(
        f"chain {args.chain} seed {seed}: warmup {args.warmup}, samples {args.samples}; "
        f"mass {args.mass or 'identity'}; checkpoint every {CKPT_EVERY} steps; "
        f"order {ALL_NAMES_SORTED}",
        flush=True,
    )

    t0 = time.perf_counter() - wall_prev

    def save(st, done: bool = False):
        with open(state_file, "wb") as f:
            pickle.dump(jax.device_get(st), f)
        if samples:
            np.savez(
                npz_path,
                **{k: np.stack([s[k] for s in samples]) for k in samples[0]},
                **{f"_{k}": np.stack([e[k] for e in extras]) for k in extras[0]},
            )
        n = len(samples)
        meta_path.write_text(
            json.dumps(
                {
                    "config": cfg,
                    "chain": args.chain,
                    "seed": seed,
                    "step": int(st.i),
                    "warmup": args.warmup,
                    "samples": n,
                    "target": args.samples,
                    "wall_s": time.perf_counter() - t0,
                    "step_size": float(st.adapt_state.step_size),
                    "mean_num_steps": float(np.mean([e["num_steps"] for e in extras]))
                    if extras
                    else float("nan"),
                    "divergences": int(np.sum([e["diverging"] for e in extras])),
                    "mean_accept": float(np.mean([e["accept_prob"] for e in extras]))
                    if extras
                    else float("nan"),
                    "done": done,
                },
                indent=2,
            )
        )

    i = int(state.i)
    t_last = time.perf_counter()
    while i < total:
        state = step(state)
        jax.block_until_ready(state.z)
        i = int(state.i)
        if i > args.warmup:
            samples.append({k: np.asarray(v) for k, v in postprocess(state.z).items()})
            extras.append(
                {
                    "num_steps": np.asarray(state.num_steps, dtype=np.int64),
                    "diverging": np.asarray(state.diverging, dtype=np.int8),
                    "accept_prob": np.asarray(state.accept_prob, dtype=np.float64),
                    "potential_energy": np.asarray(
                        state.potential_energy, dtype=np.float64
                    ),
                }
            )
        if i % CKPT_EVERY == 0 or i == total:
            save(state)
            now = time.perf_counter()
            phase = "warmup" if i <= args.warmup else "sample"
            print(
                f"chain {args.chain} {phase} {i}/{total}  "
                f"{(now - t_last) / CKPT_EVERY:.0f} s/it, {int(state.num_steps)} steps "
                f"of size {float(state.adapt_state.step_size):.3g}, "
                f"accept {float(state.accept_prob):.2f}, "
                f"{len(samples)} samples, {(now - t0) / 3600:.2f} h",
                flush=True,
            )
            t_last = now

    save(state, done=True)
    n_div = int(np.sum([e["diverging"] for e in extras]))
    print(
        f"chain {args.chain} done: {len(samples)} samples, {n_div} divergences, "
        f"mean accept {np.mean([e['accept_prob'] for e in extras]):.3f}, "
        f"{(time.perf_counter() - t0) / 3600:.2f} h",
        flush=True,
    )


if __name__ == "__main__":
    main()
