"""NUTS on the monolith: one chain per process, blocks of samples saved as they land.

The model is `calibrate.model_real` on the monolith Tesseract; every
leapfrog step costs one forward-mode Jacobian over the 14 physics latents
(~2.5 s on the real-cook grid). Initialised at the Gauss-Newton MAP, dense
mass matrix seeded from the Fisher covariance (noise block from the
finite-difference Hessian of run_laplace.py) and adapted during warmup.

Run from app/:  uv run python -u ../scripts/run_nuts_mono.py --chain 0
Writes data/calibration/nuts_mono_real/chain{c}.npz (+ meta json) after
every block; merge with scripts/merge_nuts_mono.py.
"""

import argparse
import json
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
    NAMES,
    NOISE_NAMES,
    fisher_inverse_mass,
)
from numpyro.infer import MCMC, NUTS, init_to_value

REPO = Path(__file__).resolve().parents[1]
EXTRA = ("num_steps", "diverging", "accept_prob")


def main():
    """Run one NUTS chain on the monolith, saving samples block by block."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None, help="default: chain index")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--block", type=int, default=25, help="samples per saved block")
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

    gn = json.loads((cal / args.init_physics).read_text())
    noise = json.loads((cal / args.init_noise).read_text())
    init = {n: float(gn["x"][n]) for n in NAMES} if "x" in gn else {}
    if not init:  # a plain MAP json
        init = {n: float(gn[n]) for n in NAMES[:7]}
        init.update(dict(zip(NAMES[7:], phi_full(gn), strict=True)))
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
    n_blocks = max(1, -(-args.samples // args.block))
    mcmc = MCMC(
        kernel, num_warmup=args.warmup, num_samples=args.block, progress_bar=True
    )
    print(
        f"chain {args.chain} seed {seed}: warmup {args.warmup}, samples {args.samples} in "
        f"{n_blocks} blocks of {args.block}; mass {args.mass or 'identity'}; "
        f"order {ALL_NAMES_SORTED}",
        flush=True,
    )
    t0 = time.perf_counter()
    samples, extras = {}, {}
    meta_path = out_dir / f"chain{args.chain}.json"
    for b in range(n_blocks):
        if b == 0:
            mcmc.run(jax.random.PRNGKey(seed), extra_fields=EXTRA)
        else:
            mcmc.post_warmup_state = mcmc.last_state
            mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=EXTRA)
        s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        e = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}
        samples = {
            k: np.concatenate([samples[k], v]) if k in samples else v
            for k, v in s.items()
        }
        extras = {
            k: np.concatenate([extras[k], v]) if k in extras else v
            for k, v in e.items()
        }
        n = len(next(iter(samples.values())))
        np.savez(
            out_dir / f"chain{args.chain}.npz",
            **samples,
            **{f"_{k}": v for k, v in extras.items()},
        )
        wall = time.perf_counter() - t0
        meta_path.write_text(
            json.dumps(
                {
                    "chain": args.chain,
                    "seed": seed,
                    "warmup": args.warmup,
                    "samples": n,
                    "target": args.samples,
                    "wall_s": wall,
                    "step_size": float(mcmc.last_state.adapt_state.step_size),
                    "mean_num_steps": float(np.mean(extras["num_steps"])),
                    "divergences": int(np.sum(extras["diverging"])),
                    "mean_accept": float(np.mean(extras["accept_prob"])),
                    "done": n >= args.samples,
                },
                indent=2,
            )
        )
        print(
            f"  block {b + 1}/{n_blocks}: {n} samples, {wall / 60:.1f} min, step "
            f"{float(mcmc.last_state.adapt_state.step_size):.3g}, mean steps "
            f"{np.mean(e['num_steps']):.1f}, accept {np.mean(e['accept_prob']):.2f}, "
            f"divergent {int(np.sum(extras['diverging']))}",
            flush=True,
        )
    print(
        f"chain {args.chain} done: {n} samples in {(time.perf_counter() - t0) / 3600:.2f} h"
    )


if __name__ == "__main__":
    main()
