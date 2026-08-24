"""Roadmap stretch item 1: cook time as a decision variable.

The perfect brisket defines its own time: an outer scan over planned
durations under a pure quality objective (target doneness, bark
constraint, maximum retained moisture) finds the duration perfection
takes. The serving deadline then positions the *start*: among durations at
or beyond the quality optimum, the chance layer (NUTS-real posterior
draws) finds the shortest planned duration with P(serveably done) >= 95%
- light the fire that many hours before dinner; the hold absorbs the
margin.

Run from app/:  uv run python -u ../scripts/run_timeplan.py
Writes figures/timeplan.png + data/control/timeplan_result.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "src"))

from brisket_app.calibrate import PHI_SITES, PHI_SITES_IGNITION, THETA_SITES, phi_full
from brisket_app.clients import meat_client, pit_client
from brisket_app.timeplan import chance_of_done, optimal_duration, plan_duration

REPO = Path(__file__).resolve().parents[1]
DURATIONS_H = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
N_DRAWS = 16
C_SERVEABLE = 0.85
P_TARGET = 0.95
ICS = {"tpit0": 293.15, "t_amb": 300.0, "t0_meat": 281.15}


def main():
    """Pick the cook duration by quality and the start time by uncertainty.

    Scans DURATIONS_H at the MAP-real parameters (inner 60-iteration
    schedule optimization per duration) to find the duration quality
    prefers, then draws N_DRAWS NUTS-real posterior samples and finds the
    shortest duration at or beyond the quality optimum with
    P(serveably done) >= P_TARGET — or reports an honest refusal when no
    duration certifies. Writes data/control/timeplan_result.json and
    figures/timeplan.png: the quality objective versus duration (annotated
    with retained moisture), and per-duration doneness draws with
    P(serveably done) and the chance-constrained start time.
    """
    m = json.loads((REPO / "data" / "calibration" / "map_real.json").read_text())
    theta = np.array([m[n] for n, _ in THETA_SITES])
    phi = phi_full(m)
    meat_tx, pit_tx = meat_client(), pit_client()

    print("quality scan over durations (inner: 60-iter asymmetric+juicy opt)")
    records = plan_duration(theta, phi, meat_tx, pit_tx, DURATIONS_H, ics=ICS)
    best = optimal_duration(records)
    t_star = best["t_f_h"]
    print(
        f"perfection takes: {t_star:.0f} h "
        f"(J={best['j_quality']:.3f}, doneness={best['doneness']:.3f}, "
        f"juiciness={best['juiciness']:.3f})"
    )

    print(f"empirical done-fraction from {N_DRAWS} NUTS-real posterior draws")
    post_path = REPO / "data" / "calibration" / "nuts_samples_real.npz"
    post = dict(np.load(post_path))
    missing = [n for n, _ in THETA_SITES + PHI_SITES + PHI_SITES_IGNITION if n not in post]
    if missing:
        raise SystemExit(
            f"{post_path.name} is missing sites {missing}; "
            "regenerate NUTS under the current parameter schema"
        )
    n_total = len(np.asarray(post[THETA_SITES[0][0]]).reshape(-1))
    rng = np.random.default_rng(0)
    idx = rng.choice(n_total, size=min(N_DRAWS, n_total), replace=False)
    thetas = np.stack(
        [
            [float(np.asarray(post[n]).reshape(-1)[i]) for n, _ in THETA_SITES]
            for i in idx
        ]
    )
    phis = np.stack(
        [
            [
                float(np.asarray(post[n]).reshape(-1)[i])
                for n, _ in PHI_SITES + PHI_SITES_IGNITION
            ]
            for i in idx
        ]
    )

    ps, draws = [], []
    for rec in records:
        p, done = chance_of_done(
            rec, thetas, phis, meat_tx, pit_tx, c_serveable=C_SERVEABLE, ics=ICS
        )
        ps.append(p)
        draws.append(done)

    t_fs = np.array([r["t_f_h"] for r in records])
    ok = [i for i in range(len(t_fs)) if t_fs[i] >= t_star and ps[i] >= P_TARGET]
    if ok:
        i_plan = ok[0]
        t_plan = float(t_fs[i_plan])
        print(
            f"shortest scanned duration at/above quality opt with empirical "
            f"P(done)>={P_TARGET:.2f} over {len(idx)} draws: {t_plan:.0f} h "
            f"(P={ps[i_plan]:.2f}; not a 95% certification)"
        )
    else:
        i_plan, t_plan = None, None
        print(
            "chance layer: no scanned duration reaches the empirical "
            "P_target over these draws — not a certification of infeasibility"
        )

    (REPO / "data" / "control").mkdir(parents=True, exist_ok=True)
    (REPO / "data" / "control" / "timeplan_result.json").write_text(
        json.dumps(
            {
                "durations_h": t_fs.tolist(),
                "j_quality": [r["j_quality"] for r in records],
                "doneness_map": [r["doneness"] for r in records],
                "w_retained": [r["w_retained"] for r in records],
                "p_done": ps,
                "c_serveable": C_SERVEABLE,
                "p_target": P_TARGET,
                "t_f_quality_h": t_star,
                "t_f_planned_h": t_plan,
                "margin_h": (t_plan - t_star) if t_plan is not None else None,
                "n_draws": len(idx),
                "schedules": {str(r["t_f_h"]): r["u_opt"].tolist() for r in records},
                "doneness_draws": {
                    str(t_fs[i]): draws[i].tolist() for i in range(len(t_fs))
                },
            },
            indent=2,
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, MUTED = "#256abf", "#eb6834", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.plot(
        t_fs,
        [r["j_quality"] for r in records],
        "o-",
        color=BLUE,
        lw=2,
        label="quality objective J*(t_f)",
    )
    ax.axvline(t_star, color=ORANGE, lw=1.5, ls=":")
    y_ann = float(np.max([r["j_quality"] for r in records]))
    ax.annotate(
        f"perfection takes {t_star:.0f} h",
        (t_star + 0.1, y_ann),
        color=ORANGE,
        fontsize=10,
    )
    for r in records:
        ax.annotate(
            f"{r['w_retained']:.2f}",
            (r["t_f_h"], r["j_quality"]),
            textcoords="offset points",
            xytext=(0, -14),
            color=MUTED,
            fontsize=7,
            ha="center",
        )
    ax.annotate(
        "gray: retained moisture at pull",
        (0.02, 0.03),
        xycoords="axes fraction",
        color=MUTED,
        fontsize=8,
    )
    ax.set(
        xlabel="planned cook duration t_f [h]",
        ylabel="objective",
        title="the perfect brisket defines its own time (MAP smoker)",
    )
    ax.legend(fontsize=9)

    ax = axes[1]  # doneness draws and P(done) share the same [0,1] scale
    for i, d in enumerate(draws):
        ax.plot(
            [t_fs[i]] * len(d),
            d,
            ".",
            color=BLUE,
            alpha=0.35,
            ms=5,
            zorder=1,
            label="doneness draws" if i == 0 else None,
        )
    ax.axhline(C_SERVEABLE, color=MUTED, lw=1, ls="--")
    ax.annotate(
        f"serveable (C={C_SERVEABLE})",
        (t_fs[0], C_SERVEABLE + 0.008),
        color=MUTED,
        fontsize=8,
    )
    ax.plot(t_fs, ps, "o-", color=ORANGE, lw=2, label="P(serveably done)")
    ax.axhline(P_TARGET, color=ORANGE, lw=1, ls=":", alpha=0.6)
    if t_plan is not None:
        ax.axvline(t_plan, color=ORANGE, lw=1.5, ls=":")
        ax.annotate(
            f"light {t_plan:.0f} h before dinner\n"
            f"(margin {t_plan - t_star:.0f} h → the hold)",
            (t_plan + 0.1, 0.45),
            color=ORANGE,
            fontsize=9,
        )
        title = f"start-time margin over {len(idx)} posterior draws"
    else:
        ax.annotate(
            "the posterior refuses to certify ANY\n"
            "open-loop start: the closed-loop cook log\n"
            "never exercised damper ≳ 0.35, so the\n"
            "control-relevant pit response is\n"
            "unidentified — same damper input gives\n"
            "MAP pit 202 °C vs posterior-draw 118 °C.\n"
            "The fix is a designed experiment: the\n"
            "open-loop characterization burn.",
            (0.30, 0.35),
            xycoords="axes fraction",
            color=INK,
            fontsize=8.5,
            bbox={"boxstyle": "round", "fc": "#fcfcfb", "ec": "#c3c2b7"},
        )
        title = f"chance layer over {len(idx)} posterior draws: an honest refusal"
    ax.set(
        xlabel="planned cook duration t_f [h]",
        ylabel="doneness at t_f  /  probability",
        ylim=(-0.02, 1.05),
        title=title,
    )
    ax.legend(fontsize=9, loc="center right")

    fig.suptitle(
        "Cook time as a decision variable: quality picks the "
        "duration, uncertainty picks the start",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "timeplan.png", dpi=140, bbox_inches="tight")
    print("wrote figures/timeplan.png")


if __name__ == "__main__":
    main()
