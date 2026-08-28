"""Capstone stage 1: MAP fit of the coupled model to the REAL cook log.

Run from app/:  uv run python -u ../scripts/run_map_real.py [--steps N]
Writes data/calibration/map_real.json + figures/cooklog_fit_map.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

import jax
import jax.numpy as jnp
import numpyro
from brisket_app import calibrate, cooklog
from brisket_app.clients import monolith_client
from brisket_app.coupled import HR, CookGrids, check_picard, coupled_cook
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta

REPO = Path(__file__).resolve().parents[1]


def main():
    """MAP-fit the coupled model to the real cook log and plot the fit.

    Runs SVI (AutoDelta, Adam) on the real-cook model, optionally
    warm-starting the guide from a --init JSON of parameter values, and
    writes the fitted parameters to data/calibration/map_real.json. Then
    re-simulates the cook at the MAP with the measured fan schedule and
    writes figures/cooklog_fit_map.png overlaying the model pit and core
    temperatures on the observations.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument(
        "--init",
        type=str,
        default=None,
        help="JSON of parameter values to warm-start the guide",
    )
    args = ap.parse_args()

    log = cooklog.load(dt_min=10.0)
    tx = monolith_client()
    m = calibrate.bound_model_real(log, tx)

    if args.init:
        init_vals = {
            k: float(v) for k, v in json.loads(Path(args.init).read_text()).items()
        }
        guide = AutoDelta(m, init_loc_fn=numpyro.infer.init_to_value(values=init_vals))
    else:
        guide = AutoDelta(m)
    svi = SVI(m, guide, numpyro.optim.Adam(args.lr), Trace_ELBO())
    result = svi.run(jax.random.PRNGKey(0), args.steps, progress_bar=True)
    params = {k: float(np.asarray(v)) for k, v in guide.median(result.params).items()}

    out = REPO / "data" / "calibration" / "map_real.json"
    out.write_text(json.dumps(params, indent=2))
    print(f"final loss {float(result.losses[-1]):.1f}; wrote {out.name}")
    for k, v in params.items():
        print(f"  {k:>12} = {v:.5g}")

    # fit overlay figure
    theta = np.array([params[n] for n, _ in calibrate.THETA_SITES])
    phi = calibrate.phi_full(params)
    grids = CookGrids(
        t_end=float(log["t"][-1]),
        dt_exchange=float(log["t"][1] - log["t"][0]),
        n_u=len(log["t"]),
    )
    ics = {
        "tpit0": log["meta"]["tpit0_K"],
        "t_amb": 300.0,
        "t0_meat": log["meta"]["tcore0_K"],
    }
    sim = coupled_cook(
        jnp.asarray(theta),
        jnp.asarray(phi),
        jnp.asarray(log["u"]),
        tx,
        grids,
        ics=ics,
    )
    check_picard(sim)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = log["t"] / HR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, s, o, mk, name in [
        (axes[0], np.asarray(sim["tpit"]), log["obs_pit"], log["mask_pit"], "pit"),
        (axes[1], np.asarray(sim["tcore"]), log["obs_core"], log["mask_core"], "core"),
    ]:
        ax.plot(
            th[mk],
            o[mk] - 273.15,
            ".",
            ms=2.5,
            color="k",
            alpha=0.5,
            label="observed (Mache/TVWBB)",
        )
        ax.plot(th, s - 273.15, color="tab:orange", label="model @ MAP")
        ax.set(xlabel="time [h]", ylabel="temperature [C]", title=name)
        ax.legend(fontsize=8)
    fig.suptitle("Real cook - MAP calibration fit")
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "cooklog_fit_map.png", dpi=140)
    print("wrote figures/cooklog_fit_map.png")


if __name__ == "__main__":
    main()
