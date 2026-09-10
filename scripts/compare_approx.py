"""Score every approximation that has samples against the NUTS reference.

Per physics site: posterior sd in z, and the 1-Wasserstein distance to the
NUTS draws in z (quantile form — sees mean shifts and asymmetric tails an
sd comparison cannot). Run metadata (Pareto-khat, importance ESS, wall time)
is pulled from each method's result json where present.

Run from app/:  uv run python -u ../scripts/compare_approx.py
Writes figures/calibration/posterior_widths.png (sd ratio to NUTS; same path the NUTS
merge uses) and data/calibration/approx_summary.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app.approx import IGNITION_NAMES, w1_z_columns, z_matrix_physics
from brisket_app.monolith import NAMES
from brisket_app.plots import posterior_width_plot

REPO = Path(__file__).resolve().parents[1]

# label -> (samples npz, result json with run metadata)
FILES = (
    ("NUTS-mono", "nuts_mono_samples_real.npz", None),
    ("Laplace-GN", "laplace_samples_real.npz", None),
    ("TMB-ign", "tmb_ignition_samples_real.npz", "tmb_ignition_real.json"),
    ("TMB-PSIS", "psis_tmb_samples_real.npz", "psis_tmb_real.json"),
    ("Laplace-PSIS", "psis_laplace_samples_real.npz", "psis_laplace_real.json"),
)


def _meta(cal: Path, jname: str | None) -> dict:
    """Khat / ESS / wall seconds from a method's result json, where present."""
    if jname is None or not (cal / jname).exists():
        return {}
    d = json.loads((cal / jname).read_text())
    out = {}
    if d.get("khat") is not None:
        out["khat"] = float(d["khat"])
    if d.get("ess") is not None:
        out["ess"] = float(d["ess"])
    for k in ("wall_s", "wall_s_probes"):
        if d.get(k) is not None:
            out["wall_s"] = out.get("wall_s", 0.0) + float(d[k])
    return out


def main():
    """Print sd(z) and W1-vs-NUTS tables, refresh the figure and the summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    cal = REPO / "data" / "calibration"
    cols, zmats, metas = {}, {}, {}
    gn = cal / "gn_map_real.json"
    if gn.exists():
        cols["Fisher"] = json.loads(gn.read_text())["sd_z_fisher"]
    for label, fname, jname in FILES:
        p = cal / fname
        if not p.exists():
            continue
        samples = dict(np.load(p))
        zmats[label] = z_matrix_physics(samples)
        cols[label] = dict(
            zip(NAMES, np.std(zmats[label], axis=0), strict=True)
        )
        metas[label] = _meta(cal, jname)
    if not cols:
        raise SystemExit("no approximation files under data/calibration")

    print("posterior sd(z) per site")
    print(f"{'site':10s}" + "".join(f"{c:>14s}" for c in cols))
    for n in NAMES:
        print(f"{n:10s}" + "".join(f"{cols[c][n]:14.3f}" for c in cols))

    w1 = {}
    if "NUTS-mono" in zmats:
        ref = zmats["NUTS-mono"]
        for label, Z in zmats.items():
            if label == "NUTS-mono":
                continue
            w1[label] = dict(
                zip(NAMES, map(float, w1_z_columns(Z, ref)), strict=True)
            )
        print("\nW1 to NUTS in z per site (0 = matches; sees shifts and tails)")
        print(f"{'site':10s}" + "".join(f"{c:>14s}" for c in w1))
        for n in NAMES:
            print(f"{n:10s}" + "".join(f"{w1[c][n]:14.3f}" for c in w1))

    lines = [
        "# Approximation scoreboard (vs 2-chain NUTS reference)",
        "",
        (
            "W1 = per-site 1-Wasserstein to NUTS in unconstrained z"
            " (mean over sites / worst site / ignition pair);"
            " sd ratio = approx/NUTS ignition widths."
        ),
        "",
        (
            "| method | W1 mean | W1 max (site) | W1 a0 | W1 k_spread |"
            " sd ratio a0 | sd ratio k_spread | khat | IS ESS | wall |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    nuts_sd = cols.get("NUTS-mono")
    order = sorted(
        w1, key=lambda c: float(np.mean([w1[c][n] for n in NAMES]))
    )
    for label in order:
        v = w1[label]
        worst = max(NAMES, key=lambda n: v[n])
        m = metas.get(label, {})
        ratio = {
            n: cols[label][n] / nuts_sd[n] if nuts_sd else float("nan")
            for n in IGNITION_NAMES
        }
        wall = m.get("wall_s")
        wall_str = f"{wall / 60:.0f} min" if wall else "—"
        khat_str = f"{m['khat']:.2f}" if "khat" in m else "—"
        ess_str = f"{m['ess']:.0f}" if "ess" in m else "—"
        lines.append(
            f"| {label} | {np.mean([v[n] for n in NAMES]):.3f} "
            f"| {v[worst]:.3f} ({worst}) | {v['a0']:.3f} | {v['k_spread']:.3f} "
            f"| {ratio['a0']:.2f} | {ratio['k_spread']:.2f} "
            f"| {khat_str} | {ess_str} | {wall_str} |"
        )
    if w1:
        out_md = cal / "approx_summary.md"
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {out_md.relative_to(REPO)}")

    if not args.no_figure:
        posterior_width_plot(cols, REPO / "figures" / "calibration" / "posterior_widths.png")
        print("wrote figures/calibration/posterior_widths.png")


if __name__ == "__main__":
    main()
