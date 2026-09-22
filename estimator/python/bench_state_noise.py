"""The filter's state process noise, earned on the bench and written to a file, reproducibly.

The filter (estimator/src/sequential.jl) carries the plant's state in its belief with a random-walk process noise
per field group (StateNoise: a rate per square-root hour and an initial width for the temperature field, the
water, the collagen, fat and salt, the bark and ring, the pit's scalars, the ignited fraction). The rates are
not fit to any reading; they are earned by consistency against the truth on the drifting bench
(estimator/scripts/coverage_filter.jl: cooks with truths drawn from the pilot's prior, walked by drift.jl, read by the
real-cook instruments), and this driver runs that protocol end to end so it can be rerun whenever the physics
or the filter change:

  1. the tuning round: the bench on its seeds with the rates as they stand (the file's, or the code's defaults
     when there is none), the belief's state scored group by group against the plant's (the normalized
     estimation error squared per degree of freedom, NEES/dof);
  2. one covariance-matching correction: each group's rate and initial width scaled by the square root of
     its pooled NEES/dof, the standard fixed-point step, applied once, no rounds by eye. The step is full
     when it widens the belief and half (in the log) when it narrows it, both capped: a width that is only
     partly the rate's overshoots under the full step, and overshooting into overconfidence is the failure
     that matters, while overshooting into width is benign;
  3. the fresh round: the bench again on seeds it has never seen with the corrected rates; the consistency
     figure and the coverage figure are rendered from this round only, and it is the round the numbers are
     quoted from. Anything still out of band is reported, not tuned away.

The file data/calibration/state_noise.json carries the rates, the fingerprint of the physics and the filter
they were earned on (BrisketEstimator.physics_fingerprint; every driver refuses the file when the tree has moved past it, and the
tune round alone starts from stale rates),
the git commit, and both rounds' numbers. The bench is a floor: its plant is the model's own physics, so on a
real pit, where the model is wrong in ways the mock never is, the same rates leave the belief overconfident;
on real logs the check is NIS and whiteness only.

Run from app/:  uv run python ../estimator/python/bench_state_noise.py [--cooks 9] [--workers 2] [--hours 14]
                    [--prior prior_next_cook.json] [--tune-seed0 1] [--fresh-seed0 10] [--cap 4]
                    [--start file|default] [--rounds tune,fresh] [--smoke]
  --smoke: one worker, one cook, at most two hours, into scratch names (the driver's own check; nothing quoted from it)
  --rounds fresh: the fresh round only, with the file's rates as they are (the check without the correction)
  --reuse: aggregate a round's records when they all exist instead of running its workers
  --variant NAME: a bench variant (records variant_NAME_<round>_<seed>.json, figures tagged _NAME, the rates file untouched)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
RATES_FILE = REPO / "data" / "calibration" / "state_noise.json"
GROUPS = ["T", "W", "C", "F", "S", "bark", "T_pit", "m_fuel", "m_pan", "T_pan", "T_fb", "ignited"]
UNITS = {"T": "K", "W": "kg/kg", "C": "-", "F": "-", "S": "-", "bark": "-", "T_pit": "K", "m_fuel": "kg", "m_pan": "kg", "T_pan": "K", "T_fb": "K", "ignited": "-"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_workers(label: str, seed0s: list[int], n_per: int, hours: float, prior: str, sn_file: str, out_names: list[str], log_dir: Path, stale_ok: bool = False) -> list[Path]:
    """The bench workers in parallel, one Julia process per seed block (two BLAS threads each: three workers with the default
    threads ran this laptop out of memory); their logs in log_dir. `stale_ok` lets the workers start from a state-noise file
    whose fingerprint is not the tree's: the tune round only."""
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "2"}
    procs = []
    for s, out in zip(seed0s, out_names):
        log = open(log_dir / f"{label}_{s}.log", "w", encoding="utf-8")
        cmd = ["julia", "--project=estimator", "estimator/scripts/coverage_filter.jl", prior, str(n_per), str(s), out, str(hours), "0", sn_file, "64", "1" if stale_ok else "0"]
        print(f"  {label}: seeds {s}..{s + n_per - 1} -> data/mock/{out}  ({' '.join(cmd)})")
        procs.append((s, out, subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    sys.stdout.flush()
    t0 = time.time()
    failed = []
    for s, out, p, log in procs:
        p.wait()
        log.close()
        if p.returncode != 0:
            failed.append((s, p.returncode))
    print(f"  {label}: done in {(time.time() - t0) / 60:.0f} min")
    if failed:
        raise SystemExit(f"{label}: workers failed (seed0, code): {failed}; see {log_dir}")
    return [REPO / "data" / "mock" / out for out in out_names]


def load_round(files: list[Path]) -> dict:
    """The pooled consistency numbers of a round: NEES/dof per state group, of the latents and of the temperatures,
    NIS/dof, the whiteness per channel, the rates the round ran with, and the physics fingerprint."""
    cooks, sn, fp, gd, seeds, skipped = [], None, None, None, [], []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        cooks += d["cooks"]
        sn = d["state_noise"]; fp = d.get("physics"); gd = d["group_dof"]
        seeds += [c["seed"] for c in d["cooks"]]; skipped += d.get("skipped_seeds", [])
    if not cooks:
        raise SystemExit("the round produced no cooks")
    temps = [r for c in cooks for r in c["temps"]]
    out = {"n_cooks": len(cooks), "seeds": seeds, "skipped": skipped, "n_updates": len(temps), "physics": fp, "group_dof": gd,
           "rates": {g: sn[g] for g in GROUPS}, "init_sd": {g: sn[g + "0"] for g in GROUPS}, "groups": {},
           "fields": sn.get("fields", "nodes"), "node_jitter": sn.get("node_jitter", 0.0),
           "events": {k: sn[k] for k in ("T_event", "T_spritz", "m_add") if k in sn}}
    for g in GROUPS:
        v = np.array([r["nees_g"][g] for r in temps], dtype=float)
        v = v[np.isfinite(v)]
        per_cook = [float(np.nanmean([r["nees_g"][g] for r in c["temps"]])) for c in cooks]
        out["groups"][g] = {"nees_dof": float(np.mean(v)) if len(v) else float("nan"), "n": int(len(v)),
                            "cook_median": float(np.nanmedian(per_cook)), "cook_max": float(np.nanmax(per_cook))}
    nz = np.array([r["nees_z16"] for r in temps], dtype=float); nT = np.array([r["nees_T"] for r in temps], dtype=float)
    out["latents_per_cook"] = [float(np.nanmean([r["nees_z16"] for r in c["temps"]]) / 16) for c in cooks]
    out["temps_per_cook"] = [float(np.nanmean([r["nees_T"] for r in c["temps"]]) / 3) for c in cooks]
    nis = np.array([r["nis"] for r in temps if r["dof"] > 0], dtype=float); dof = np.array([r["dof"] for r in temps if r["dof"] > 0], dtype=float)
    out["latents_nees16"] = float(np.nanmean(nz) / 16); out["temps_nees3"] = float(np.nanmean(nT) / 3)
    out["nis_dof"] = float(np.nansum(nis) / np.sum(dof)) if len(dof) else float("nan")
    # the whiteness per channel: the standardized innovations' mean and lag-one autocorrelation, per cook then pooled
    white = {}
    for ch, lab in {"1": "pit probe", "2": "core probe", "3": "humidity", "4": "scale"}.items():
        seqs = [np.array([v for r in c["temps"] for v in r["innov"].get(ch, [])]) for c in cooks]
        seqs = [s for s in seqs if len(s) > 3]
        allv = np.concatenate(seqs) if seqs else np.array([])
        r1 = [np.corrcoef(s[:-1], s[1:])[0, 1] for s in seqs]
        white[lab] = {"n": int(len(allv)), "mean": float(np.mean(allv)) if len(allv) else float("nan"), "sd": float(np.std(allv, ddof=1)) if len(allv) > 1 else float("nan"),
                      "lag1": float(np.mean(r1)) if r1 else float("nan"), "band": float(2 / np.sqrt(np.mean([len(s) for s in seqs]))) if seqs else float("nan")}
    out["whiteness"] = white
    return out


def band(n_eff: float, dof: int) -> tuple[float, float]:
    """The 95 % band of a mean of chi-squared(dof)/dof variables over n_eff independent samples. The updates along a
    cook are correlated, so the count is one sample per cook-hour, not one per update; the band is indicative,
    it orders the groups rather than judging them alone."""
    k = max(int(round(n_eff * dof)), 1)
    return float(stats.chi2.ppf(0.025, k) / k), float(stats.chi2.ppf(0.975, k) / k)


def print_round(name: str, r: dict, hours: float) -> None:
    gd = r["group_dof"]; n_eff = r["n_cooks"] * hours
    print(f"{name}: {r['n_cooks']} cooks (seeds {r['seeds']}), {r['n_updates']} updates; physics {r['physics']}; the fields' noise by {r['fields']}")
    print(f"  NIS/dof {r['nis_dof']:.2f}; NEES/16 of the latents {r['latents_nees16']:.2f} (per cook: " + " ".join(f"{v:.2f}" for v in r["latents_per_cook"]) + ")")
    print(f"  NEES/3 of the temperatures {r['temps_nees3']:.2f} (per cook: " + " ".join(f"{v:.2f}" for v in r["temps_per_cook"]) + ")")
    print(f"  {'group':9s} {'rate':>10s} {'init sd':>10s} {'NEES/dof':>9s} {'cook median':>12s} {'cook max':>9s} {'band (one sample per cook-hour)':>32s}")
    for g in GROUPS:
        v = r["groups"][g]; lo, hi = band(n_eff, gd[g])
        flag = "" if lo <= v["nees_dof"] <= hi else ("  too narrow" if v["nees_dof"] > hi else "  too wide")
        print(f"  {g:9s} {r['rates'][g]:10.4g} {r['init_sd'][g]:10.4g} {v['nees_dof']:9.2f} {v['cook_median']:12.2f} {v['cook_max']:9.2f} {lo:18.2f} to {hi:.2f}{flag}")
    print("  whiteness: " + "; ".join(f"{lab} mean {w['mean']:+.2f} lag-1 {w['lag1']:+.2f} (band +-{w['band']:.2f})" for lab, w in r["whiteness"].items()))
    sys.stdout.flush()


def match(r: dict, cap: float) -> tuple[dict, dict, dict]:
    """One covariance-matching step: each group's rate and initial width scaled by the square root of its NEES/dof
    when that widens the belief, by its fourth root when it narrows it (half the step in the log: a width that is
    only partly the rate's overshoots under the full step, and overshooting into overconfidence is the failure that
    matters); the factor capped to [1/cap, cap]; a group without a finite NEES keeps its numbers."""
    rates, init, factors = {}, {}, {}
    for g in GROUPS:
        v = r["groups"][g]["nees_dof"]
        f = 1.0
        if np.isfinite(v) and v > 0:
            f = float(np.clip(np.sqrt(v) if v >= 1 else v ** 0.25, 1 / cap, cap))
        factors[g] = f; rates[g] = r["rates"][g] * f; init[g] = r["init_sd"][g] * f
    return rates, init, factors


def render(pattern: str, tag: str, log_dir: Path) -> None:
    """The consistency figure and the coverage figure from the fresh round's records."""
    for script, args in (("plot_filter_consistency.py", [pattern, tag]), ("plot_coverage.py", [pattern, "err_sd_graphfilter", "_driftgraphfilter" + tag])):
        cmd = [sys.executable, str(REPO / "estimator" / "python" / script)] + args
        res = subprocess.run(cmd, cwd=REPO / "app", capture_output=True, text=True)
        (log_dir / (script + ".log")).write_text(res.stdout + res.stderr, encoding="utf-8")
        print(res.stdout.strip())
        if res.returncode != 0:
            print(res.stderr.strip())
            raise SystemExit(f"{script} failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cooks", type=int, default=9); ap.add_argument("--workers", type=int, default=2); ap.add_argument("--hours", type=float, default=14.0)
    ap.add_argument("--prior", default="prior_next_cook.json"); ap.add_argument("--tune-seed0", type=int, default=1); ap.add_argument("--fresh-seed0", type=int, default=10)
    ap.add_argument("--cap", type=float, default=4.0, help="the most a rate moves in the one correction, as a factor either way")
    ap.add_argument("--start", choices=["file", "default"], default="file", help="the tuning round's rates: the file's (if any) or the code's defaults")
    ap.add_argument("--rounds", default="tune,fresh"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="aggregate a round's records if they all exist instead of running its workers (a rerun of the correction or the figures)")
    ap.add_argument("--variant", default="", help="a named variant: its own records and figures, the rates file untouched")
    a = ap.parse_args()
    rounds = a.rounds.split(",")
    if a.smoke:
        a.cooks, a.workers, a.hours = 1, 1, min(a.hours, 2.0)
    n_per = -(-a.cooks // a.workers)
    tune_seeds = [a.tune_seed0 + i * n_per for i in range(a.workers)]
    fresh_seeds = [a.fresh_seed0 + i * n_per for i in range(a.workers)]
    stem_tune, stem_fresh = ("bench_smoke_tune", "bench_smoke_fresh") if a.smoke else ("coverage_tune", "coverage_filter")
    if a.variant:
        stem_tune, stem_fresh = f"variant_{a.variant}_tune", f"variant_{a.variant}_filter"   # a stem the main figures' glob does not match
    rates_out = (REPO / "scratchpad" / "state_noise_smoke.json") if a.smoke else (REPO / "scratchpad" / f"state_noise_{a.variant}.json" if a.variant else RATES_FILE)
    log_dir = REPO / "scratchpad" / ("bench_state_noise_smoke" if a.smoke else f"bench_state_noise_{a.variant}" if a.variant else "bench_state_noise")
    log_dir.mkdir(parents=True, exist_ok=True)
    start_file = "default" if (a.start == "default" or not RATES_FILE.exists()) else RATES_FILE.name
    print(f"the state-noise bench: {a.cooks} cooks per round in {a.workers} workers, {a.hours:g} h each, prior {a.prior}; rounds {rounds}; "
          f"the tuning round starts from {'the code defaults' if start_file == 'default' else 'data/calibration/' + start_file}; logs in {log_dir}")
    sys.stdout.flush()
    if "tune" in rounds:
        names = [f"{stem_tune}_{s}.json" for s in tune_seeds]
        if a.reuse and all((REPO / "data" / "mock" / nm).exists() for nm in names):
            files = [REPO / "data" / "mock" / nm for nm in names]; print("  tune: reusing the records " + ", ".join(names))
        else:
            files = run_workers("tune", tune_seeds, n_per, a.hours, a.prior, start_file, names, log_dir, stale_ok=start_file != "default")
        tune = load_round(files)
        print_round("the tuning round (before the correction)", tune, a.hours)
        rates, init, factors = match(tune, a.cap)
        print("  the correction, each group's factor: " + ", ".join(f"{g} x{factors[g]:.2f}" for g in GROUPS))
        record = {"rates": rates, "init_sd": init, "fields": tune["fields"], "node_jitter": tune["node_jitter"], "events": tune["events"], "units": UNITS,
                  "physics": {"fingerprint": tune["physics"], "commit": git_commit(), "date": dt.date.today().isoformat(),
                              "what": "BrisketEstimator.physics_fingerprint: src/**/*.jl and the filter's own files; every driver refuses the file when the tree's differs, the tune round alone starts from stale rates"},
                  "method": "one covariance-matching step on the tuning round: rate and initial width scaled per group by sqrt(NEES/dof) when that widens the belief "
                            "and by (NEES/dof)^(1/4) when it narrows it, capped; the fresh round is the check and the round the numbers are quoted from; "
                            "the bench's plant is the model's own physics, so the rates are a floor",
                  "bench": {"cooks_per_round": a.cooks, "hours": a.hours, "prior": a.prior, "cap": a.cap,
                            "tune": {"seeds": tune["seeds"], "skipped": tune["skipped"], "records": [f.name for f in files], "started_from": start_file,
                                     "physics": tune["physics"], "stale_ok": start_file != "default",
                                     "rates": tune["rates"], "init_sd": tune["init_sd"], "factors": factors,
                                     "nees_dof": {g: tune["groups"][g]["nees_dof"] for g in GROUPS},
                                     "nees_dof_cook_median": {g: tune["groups"][g]["cook_median"] for g in GROUPS}, "nees_dof_cook_max": {g: tune["groups"][g]["cook_max"] for g in GROUPS},
                                     "nis_dof": tune["nis_dof"], "latents_nees16": tune["latents_nees16"], "latents_per_cook": tune["latents_per_cook"],
                                     "temps_nees3": tune["temps_nees3"], "temps_per_cook": tune["temps_per_cook"], "whiteness": tune["whiteness"]}}}
        rates_out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"  wrote {rates_out}")
        sys.stdout.flush()
    if "fresh" in rounds:
        if a.variant and not rates_out.exists():
            rates_out = RATES_FILE                                                    # a variant's fresh round runs on the rates as earned
        if not rates_out.exists():
            raise SystemExit(f"no rates file at {rates_out}: run the tuning round first")
        sn_file = rates_out.name if rates_out == RATES_FILE else str(rates_out)      # the worker takes a name in data/calibration or an absolute path
        names = [f"{stem_fresh}_{s}.json" for s in fresh_seeds]
        if a.reuse and all((REPO / "data" / "mock" / nm).exists() for nm in names):
            files = [REPO / "data" / "mock" / nm for nm in names]; print("  fresh: reusing the records " + ", ".join(names))
        else:
            files = run_workers("fresh", fresh_seeds, n_per, a.hours, a.prior, sn_file, names, log_dir)          # strict: the rates must be this tree's
        fresh = load_round(files)
        print_round("the fresh round (the rates as written, seeds never seen)", fresh, a.hours)
        record = json.loads(rates_out.read_text(encoding="utf-8"))
        if a.variant:                                                                 # a variant records its numbers beside the rates, never in them
            rates_out = REPO / "scratchpad" / f"state_noise_{a.variant}.json"
        record.setdefault("bench", {})["fresh"] = {"seeds": fresh["seeds"], "skipped": fresh["skipped"], "records": [f.name for f in files], "physics": fresh["physics"],
                                                   "nees_dof": {g: fresh["groups"][g]["nees_dof"] for g in GROUPS},
                                                   "nees_dof_cook_median": {g: fresh["groups"][g]["cook_median"] for g in GROUPS}, "nees_dof_cook_max": {g: fresh["groups"][g]["cook_max"] for g in GROUPS},
                                                   "nis_dof": fresh["nis_dof"], "latents_nees16": fresh["latents_nees16"], "latents_per_cook": fresh["latents_per_cook"],
                                                   "temps_nees3": fresh["temps_nees3"], "temps_per_cook": fresh["temps_per_cook"], "whiteness": fresh["whiteness"]}
        fp_fresh, fp_file = fresh["physics"], record.get("physics", {}).get("fingerprint")
        if fp_fresh != fp_file:                                                       # a stale fresh round is never written into the file
            raise SystemExit(f"the fresh round ran on physics {fp_fresh}; the rates file records {fp_file}: stale, rerun the tune round on this tree")
        rates_out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"  wrote {rates_out}")
        render(f"{stem_fresh}_*.json", "_smoke" if a.smoke else (f"_{a.variant}" if a.variant else ""), log_dir)
    print("done")


if __name__ == "__main__":
    main()
