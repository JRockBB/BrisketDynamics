"""Time the MPC objective and gradient, separating JIT from repeated work.

Run with app/.venv/Scripts/python scripts/bench_mpc.py --out <json>.
--repo can point to a source snapshot for a same-input comparison.
"""

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path


def main():
    """Compare identical objective inputs with separately reported first-call cost."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cases", nargs="+", default=["11:15:off", "16:21:off", "16:21:on", "15:20:on"]
    )
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--decisions-only", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "app/src"))
    os.environ["PYTHON_JULIAPKG_PROJECT"] = str(repo)
    os.environ["JULIA_PROJECT"] = str(repo)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from brisket_app.calibrate import THETA_SITES, phi_full
    from brisket_app.clients import monolith_client
    from brisket_app.control import ControlConfig, objective
    from brisket_app.coupled import CookGrids

    m = json.loads((root / "data/calibration/map_real.json").read_text())
    b = json.loads((root / "data/control/fullplan_result.json").read_text())
    theta = jnp.array([m[n] for n, _ in THETA_SITES])
    phi = np.array(phi_full(m))
    phi[-1] = 0.15
    phi = jnp.array(phi)
    tx = monolith_client()
    has_load = "T_load" in inspect.signature(objective).parameters
    rows = []
    for case in args.cases:
        hours, knots, mode = case.split(":")
        hours, knots = float(hours), int(knots)
        grids = CookGrids(t_end=hours * 3600, dt_exchange=600, n_u=knots)
        cc = ControlConfig(
            opt_wrap=True, **({"opt_load": mode == "on"} if has_load else {})
        )
        u = np.clip(
            np.interp(
                np.linspace(0, 1, knots),
                np.linspace(0, 1, len(b["best_u"])),
                b["best_u"],
            ),
            0.001,
            0.999,
        )
        logit = lambda x: np.log(x / (1 - x))
        zw = logit(b["best_t_wrap_h"] / b["best_t_f_h"])
        zp = logit(b["best_t_pull_h"] / b["best_t_f_h"])
        zl = logit(
            np.clip((b["best_T_load"] - 366.48) / (422.04 - 366.48), 0.001, 0.999)
        )
        variables = tuple(
            map(jnp.asarray, (logit(u), zw, zp, 6.9, 6.9, 0.0, 0.0, 0.0, 0.0, zl))
        )

        def fun(z, zw, zp, za, zs, zd, zm, zts, zda, zl, th, ph, grids=grids, cc=cc):
            return objective(
                z,
                zw,
                th,
                ph,
                tx,
                grids,
                cc,
                {"abstol": 1e-6, "reltol": 1e-6},
                ics={"tpit0": 300.0, "t_amb": 300.0},
                z_p=zp,
                z_a=za,
                z_s=zs,
                z_d=zd,
                z_m=zm,
                z_ts=zts,
                z_da=zda,
                t_salt=12 * 3600.0,
                **({"z_l": zl, "T_load": 0.0, "t_load": -1e9} if has_load else {}),
            )

        vg = jax.jit(
            jax.value_and_grad(
                fun, argnums=tuple(range(10 if args.decisions_only else 12))
            )
        )
        times = []
        for i in range(args.repeat + 1):
            t0 = time.perf_counter()
            loss, grad = jax.block_until_ready(vg(*variables, theta, phi))
            times.append(time.perf_counter() - t0)
            print(f"{case} call {i}: {times[-1]:.3f}s J={float(loss):.9g}", flush=True)
        rows.append(
            {
                "case": case,
                "first_s": times[0],
                "warm_s": times[1:],
                "loss": float(loss),
                "gradient": [np.asarray(x).tolist() for x in grad],
            }
        )
        args.out.write_text(json.dumps({"repo": str(repo), "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
