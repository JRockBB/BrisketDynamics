"""Render a pilot journal (JSONL) into the cook picture.

Top panel: measured pit vs the pushed setpoint reference, replan marks,
wrap/pull advisories, and the meat probe. Bottom panel: the ambient
belief track (with 1-sd band) vs the innovation stream — the KF finding
the weather.

Usage (from app/):
    uv run python ../scripts/plot_pilot.py ../data/control/pilot_http.jsonl \
        --truth-url http://127.0.0.1:8181   # overlay the mock's true ambient
    uv run python ../scripts/plot_pilot.py ../data/control/pilot_sim.jsonl \
        --figure ../figures/pilot_sim.png
"""

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HR = 3600.0
K2C = lambda k: np.asarray(k, dtype=float) - 273.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("journal")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--truth-url", default=None,
                    help="mock server base URL: overlay /sim/traj truth")
    args = ap.parse_args()

    ev = [json.loads(l) for l in Path(args.journal).read_text().splitlines() if l]
    segs = [e for e in ev if e["kind"] == "segment"]
    replans = [e for e in ev if e["kind"] == "replan"]
    advis = [e for e in ev if e["kind"] == "advisory"]
    assim = [e for e in ev if e["kind"] == "assimilate"]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 7.2), sharex=True, height_ratios=[2.2, 1]
    )
    first = True
    for s in segs:
        t = np.asarray(s["t"]) / HR
        ax.plot(np.asarray(s["sp_t"]) / HR, K2C(s["sp"]), color="tab:red",
                lw=1.0, alpha=0.8, label="setpoint pushed" if first else None)
        ax.plot(t, K2C(s["tpit"]), color="tab:orange", lw=1.4,
                label="pit measured" if first else None)
        tc = [v for v in s["tcore"] if v is not None]
        ax.plot(t[: len(tc)], K2C(tc), color="tab:green", lw=1.4,
                label="meat probe" if first else None)
        first = False
    for r in replans:
        ax.axvline(r["t"] / HR, color="gray", lw=0.6, alpha=0.5)
    for a in advis:
        ax.axvline(a["t"] / HR, color="k", lw=1.2, ls="--")
        ax.annotate(a["event"].upper(), (a["t"] / HR, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(3, -14), fontsize=8)
    ax.set(ylabel="temperature [C]",
           title=f"Pilot cook journal: {Path(args.journal).name} "
                 f"({len(replans)} replans)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ta = np.asarray([e["t"] for e in assim]) / HR
    bel = K2C([e["t_amb_belief"] for e in assim])
    sd = np.sqrt([e["t_amb_var"] for e in assim])
    ax2.plot(ta, bel, "o-", color="tab:blue", ms=3, label="ambient belief")
    ax2.fill_between(ta, bel - sd, bel + sd, color="tab:blue", alpha=0.15)
    if args.truth_url:
        with urllib.request.urlopen(
            args.truth_url.rstrip("/") + "/sim/traj", timeout=30
        ) as r:
            traj = json.loads(r.read().decode())
        ax2.plot(np.asarray(traj["t"]) / HR, K2C(traj["t_amb"]), color="k",
                 lw=1.0, ls=":", label="true ambient (sim)")
    ax2.set(xlabel="cook time [h]", ylabel="ambient [C]")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = args.figure or str(Path(args.journal).with_suffix(".png"))
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
