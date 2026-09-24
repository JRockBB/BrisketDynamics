"""The filter's state process noise, earned on the bench and written to a file, reproducibly.

The filter (estimator/src/sequential.jl) carries the plant's state in its belief with a random-walk process noise
per field group (StateNoise: a rate per square-root hour and an initial width for the temperature field, the
water, the collagen, fat and salt, the bark and ring, the pit's scalars, the ignited fraction). The rates are
not fit to any reading; they are identified from the filter's own innovations on the drifting bench
(estimator/scripts/coverage_filter.jl: cooks with truths drawn from the pilot's prior, walked by drift.jl, read by the
real-cook instruments) and checked against the truth there, and this driver runs that protocol end to end so it
can be rerun whenever the physics or the filter change:

  1. the tuning round: the bench on its seeds with the rates as they stand (the file's, or the code's defaults
     when there is none); each cook's filter carries the accumulator of the innovation autocovariances'
     sensitivity to every group's variance rate (estimator/src/als.jl), and the belief's state is scored group
     by group against the plant's (the normalized estimation error squared per degree of freedom, NEES/dof);
  2. one autocovariance least-squares step (estimator/python/als_fit.py; Odelson, Rajamani, Rawlings 2006 in
     the fixed-gain, time-varying form): the products of the standardized innovations at every pair of updates
     inside the lag window, less the filter's own white model, regressed on the change of the twelve variance
     rates, pooled over the cooks, bounded by the cap on each rate. A group the innovations cannot see keeps
     its numbers and is reported; the initial widths are kept. Applied once, no rounds by eye. The step needs
     no truth, so it transfers to a real cook where the earlier covariance matching against the plant's state
     could not; it also cannot turn a persistent error into white noise, which that matching did;
  3. the fresh round: the bench again on seeds it has never seen with the identified rates; the consistency
     figure and the coverage figure are rendered from this round only, and it is the round the numbers are
     quoted from. Anything still out of band is reported, not tuned away.

Each round's records are listed in a manifest (data/mock/bench_manifest_<round>.json: the records, their seed
blocks, the cook count, the physics and the state noise they ran with; estimator/python/bench_manifest.py), and
every renderer takes the manifest and refuses records that are not the round's. The cook count is whole blocks
(workers × cooks per worker); the fresh seeds start where the tuning seeds end unless given, and the two sets may
not overlap. The fit's standard errors are quoted twice: from the model's rows, and from a leave-one-cook-out
jackknife, since a cook's products are not independent of one another.

The file data/calibration/state_noise.json carries the rates, the fingerprint of the physics and the filter
they were earned on (BrisketEstimator.physics_fingerprint; every driver refuses the file when the tree has moved past it, and the
tune round alone starts from stale rates),
the git commit, and both rounds' numbers. The bench is a floor: its plant is the model's own physics, so on a
real pit, where the model is wrong in ways the mock never is, the same rates leave the belief overconfident;
on real logs the check is NIS and whiteness only.

Run from app/:  uv run python ../estimator/python/bench_state_noise.py [--cooks 10] [--workers 2] [--hours 14]
                    [--prior prior_next_cook.json] [--tune-seed0 1] [--fresh-seed0 <after the tuning seeds>] [--cap 4] [--lag-h 2]
                    [--start file|default] [--rounds tune,fresh] [--smoke]
  --smoke: one worker, one cook, at most two hours, into scratch names (the driver's own check; nothing quoted from it)
  --rounds fresh: the fresh round only, with the file's rates as they are (the check without the correction)
  --reuse: aggregate a round's records when they all exist and are the round's (its seeds, physics and rates) instead of running its workers
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from als_fit import als_fit, format_fit, format_lags, lag_table, pool
from bench_manifest import validate_records, write_manifest

REPO = Path(__file__).resolve().parents[2]
RATES_FILE = REPO / "data" / "calibration" / "state_noise.json"
GROUPS = ["T", "W", "C", "F", "S", "bark", "T_pit", "m_fuel", "m_pan", "T_pan", "T_fb", "ignited"]
UNITS = {"T": "K", "W": "kg/kg", "C": "-", "F": "-", "S": "-", "bark": "-", "T_pit": "K", "m_fuel": "kg", "m_pan": "kg", "T_pan": "K", "T_fb": "K", "ignited": "-"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_workers(label: str, seed0s: list[int], n_per: int, hours: float, prior: str, sn_file: str, out_names: list[str], log_dir: Path, stale_ok: bool = False,
                lag_h: float = 2.0) -> list[Path]:
    """The bench workers in parallel, one Julia process per seed block (two BLAS threads each: three workers with the default
    threads ran this laptop out of memory); their logs in log_dir. `stale_ok` lets the workers start from a state-noise file
    whose fingerprint is not the tree's: the tune round only."""
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "2"}
    procs = []
    for s, out in zip(seed0s, out_names):
        log = open(log_dir / f"{label}_{s}.log", "w", encoding="utf-8")
        cmd = ["julia", "--project=estimator", "estimator/scripts/coverage_filter.jl", prior, str(n_per), str(s), out, str(hours), "0", sn_file, "64", "1" if stale_ok else "0", str(lag_h)]
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
    als = [c["als"] for c in cooks if c.get("als")]
    out_als_summaries = als
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
    gaps = np.array([r["gap"] for r in temps if r.get("updated") and r.get("gap") is not None], dtype=float)   # the updates' fixed-point gaps (records since 2026-09-24)
    gaps = gaps[np.isfinite(gaps)]
    out["gap"] = {"n": int(len(gaps)), "max": float(np.max(gaps)) if len(gaps) else None, "median": float(np.median(gaps)) if len(gaps) else None,
                  "over_half_sd": int(np.sum(gaps > 0.5)) if len(gaps) else None}
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
    out["als"] = pool(als) if als else None                             # the innovation autocovariances' statistics, pooled (absent on records made without the accumulator)
    out["als_summaries"] = out_als_summaries                            # per cook, for the leave-one-cook-out jackknife
    return out


def band(n_eff: float, dof: int) -> tuple[float, float]:
    """The 95 % band of a mean of chi-squared(dof)/dof variables over n_eff independent samples. The updates along a
    cook are correlated, so the count is one sample per cook-hour, not one per update; the band is indicative,
    it orders the groups rather than judging them alone."""
    k = max(int(round(n_eff * dof)), 1)
    return float(stats.chi2.ppf(0.025, k) / k), float(stats.chi2.ppf(0.975, k) / k)


def cluster_interval(per_cook: list[float]) -> tuple[float, float]:
    """The 95 % interval of the mean over cooks from the cooks' own spread (a t interval on the per-cook values: the
    updates within a cook are not independent, the cooks are), beside which the chi-squared band of one sample per
    cook is the reference for a consistent filter."""
    v = np.array([x for x in per_cook if np.isfinite(x)], dtype=float)
    if len(v) < 3:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v))
    return float(np.mean(v) - half), float(np.mean(v) + half)


def print_round(name: str, r: dict, hours: float) -> None:
    gd = r["group_dof"]; n_eff = r["n_cooks"] * hours
    print(f"{name}: {r['n_cooks']} cooks (seeds {r['seeds']}), {r['n_updates']} updates; physics {r['physics']}; the fields' noise by {r['fields']}")
    lo16, hi16 = band(r["n_cooks"], 16); lo3, hi3 = band(r["n_cooks"], 3)
    c16 = cluster_interval(r["latents_per_cook"]); c3 = cluster_interval(r["temps_per_cook"])
    print(f"  NIS/dof {r['nis_dof']:.2f}; NEES/16 of the latents {r['latents_nees16']:.2f} (per cook: " + " ".join(f"{v:.2f}" for v in r["latents_per_cook"])
          + f"; the mean's interval from the cooks' spread {c16[0]:.2f} to {c16[1]:.2f}; a consistent filter's band for {r['n_cooks']} cooks {lo16:.2f} to {hi16:.2f})")
    print(f"  NEES/3 of the temperatures {r['temps_nees3']:.2f} (per cook: " + " ".join(f"{v:.2f}" for v in r["temps_per_cook"])
          + f"; the mean's interval {c3[0]:.2f} to {c3[1]:.2f}; the band {lo3:.2f} to {hi3:.2f})")
    g = r["gap"]
    if g["n"]:
        print(f"  the updates' fixed-point gap (the graph's mean at the final centre against the centre, whitened sds, worst coordinate): "
              f"max {g['max']:.3f}, median {g['median']:.3f}, {g['over_half_sd']} of {g['n']} updates over half an sd")
    else:
        print("  the updates' fixed-point gap: not in these records (made before 2026-09-24)")
    print(f"  {'group':9s} {'rate':>10s} {'init sd':>10s} {'NEES/dof':>9s} {'cook median':>12s} {'cook max':>9s} {'band (one sample per cook-hour)':>32s}")
    for g in GROUPS:
        v = r["groups"][g]; lo, hi = band(n_eff, gd[g])
        flag = "" if lo <= v["nees_dof"] <= hi else ("  too narrow" if v["nees_dof"] > hi else "  too wide")
        print(f"  {g:9s} {r['rates'][g]:10.4g} {r['init_sd'][g]:10.4g} {v['nees_dof']:9.2f} {v['cook_median']:12.2f} {v['cook_max']:9.2f} {lo:18.2f} to {hi:.2f}{flag}")
    print("  whiteness: " + "; ".join(f"{lab} mean {w['mean']:+.2f} lag-1 {w['lag1']:+.2f} (band +-{w['band']:.2f})" for lab, w in r["whiteness"].items()))
    sys.stdout.flush()


CHANNELS = {1: "pit probe", 2: "core probe", 3: "humidity", 4: "scale"}
LAG_BINS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60]


def identify(r: dict, cap: float) -> tuple[dict, dict, dict, dict]:
    """The autocovariance least-squares step on the round's pooled statistics: the rates per group after it, the
    initial widths as they were, the factor per group, and the fit's own record (als_fit.py)."""
    if r["als"] is None:
        raise SystemExit("the round's records carry no innovation-autocovariance statistics: rerun the workers with the accumulator (als_lag_h > 0)")
    if list(r["als"]["groups"]) != GROUPS:
        raise SystemExit(f"the accumulator's groups {r['als']['groups']} are not the driver's {GROUPS}")
    fit = als_fit(r["als"], cap=cap, summaries=r["als_summaries"])
    rates = {g: fit["rates_new"][i] for i, g in enumerate(GROUPS)}
    init = {g: r["init_sd"][g] for g in GROUPS}
    factors = {g: fit["factors"][i] for i, g in enumerate(GROUPS)}
    fit["lags"] = lag_table(r["als"], fit, CHANNELS, LAG_BINS)
    return rates, init, factors, fit


def reusable(files: list[Path], blocks: list[list[int]], physics: str | None = None, state_noise: dict | None = None) -> bool:
    """Whether existing records are the round's own (its seed blocks, one physics, the expected physics and rates when given)."""
    try:
        validate_records(files, blocks, physics=physics, state_noise=state_noise)
        return True
    except ValueError as e:
        print(f"  the existing records are not the round's ({e}): the workers run")
        return False


def render(manifest: Path, tag: str, log_dir: Path) -> None:
    """The consistency figure and the coverage figure from the fresh round's manifest."""
    for script, args in (("plot_filter_consistency.py", [str(manifest), tag]), ("plot_coverage.py", [str(manifest), "err_sd_graphfilter", "_driftgraphfilter" + tag])):
        cmd = [sys.executable, str(REPO / "estimator" / "python" / script)] + args
        res = subprocess.run(cmd, cwd=REPO / "app", capture_output=True, text=True)
        (log_dir / (script + ".log")).write_text(res.stdout + res.stderr, encoding="utf-8")
        print(res.stdout.strip())
        if res.returncode != 0:
            print(res.stderr.strip())
            raise SystemExit(f"{script} failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cooks", type=int, default=10, help="cooks per round, rounded up to whole blocks of workers")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--hours", type=float, default=14.0)
    ap.add_argument("--prior", default="prior_next_cook.json"); ap.add_argument("--tune-seed0", type=int, default=1)
    ap.add_argument("--fresh-seed0", type=int, default=None, help="the fresh round's first seed: right after the tuning round's seeds unless given; the sets may not overlap")
    ap.add_argument("--cap", type=float, default=4.0, help="the most a rate moves in the one identification step, as a factor either way")
    ap.add_argument("--lag-h", type=float, default=2.0, help="the lag window, in hours, of the innovation autocovariances each cook accumulates")
    ap.add_argument("--start", choices=["file", "default"], default="file", help="the tuning round's rates: the file's (if any) or the code's defaults")
    ap.add_argument("--rounds", default="tune,fresh"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="aggregate a round's records if they all exist instead of running its workers (a rerun of the correction or the figures)")
    ap.add_argument("--variant", default="", help="a named variant: its own records and figures, the rates file untouched")
    a = ap.parse_args()
    rounds = a.rounds.split(",")
    if a.smoke:
        a.cooks, a.workers, a.hours = 1, 1, min(a.hours, 2.0)
    n_per = -(-a.cooks // a.workers)
    n_cooks = n_per * a.workers                                                   # whole blocks: what actually runs
    a.cooks = n_cooks
    if a.fresh_seed0 is None:
        a.fresh_seed0 = a.tune_seed0 + n_cooks
    tune_seeds = [a.tune_seed0 + i * n_per for i in range(a.workers)]
    fresh_seeds = [a.fresh_seed0 + i * n_per for i in range(a.workers)]
    tune_blocks = [list(range(s, s + n_per)) for s in tune_seeds]
    fresh_blocks = [list(range(s, s + n_per)) for s in fresh_seeds]
    overlap = sorted({x for b in tune_blocks for x in b} & {x for b in fresh_blocks for x in b})
    if overlap:
        raise SystemExit(f"the tuning seeds {tune_blocks} and the fresh seeds {fresh_blocks} share {overlap}: the fresh round would not be held out")
    stem_tune, stem_fresh = ("bench_smoke_tune", "bench_smoke_fresh") if a.smoke else ("coverage_tune", "coverage_filter")
    if a.variant:
        stem_tune, stem_fresh = f"variant_{a.variant}_tune", f"variant_{a.variant}_filter"   # a stem the main figures' glob does not match
    rates_out = (REPO / "scratchpad" / "state_noise_smoke.json") if a.smoke else (REPO / "scratchpad" / f"state_noise_{a.variant}.json" if a.variant else RATES_FILE)
    tag = "_smoke" if a.smoke else (f"_{a.variant}" if a.variant else "")
    manifest_of = lambda rnd: REPO / "data" / "mock" / f"bench_manifest_{rnd}{tag}.json"   # noqa: E731
    log_dir = REPO / "scratchpad" / ("bench_state_noise_smoke" if a.smoke else f"bench_state_noise_{a.variant}" if a.variant else "bench_state_noise")
    log_dir.mkdir(parents=True, exist_ok=True)
    start_file = "default" if (a.start == "default" or not RATES_FILE.exists()) else RATES_FILE.name
    print(f"the state-noise bench: {a.cooks} cooks per round in {a.workers} workers ({n_per} each; tuning seeds {tune_blocks}, fresh seeds {fresh_blocks}), {a.hours:g} h each, prior {a.prior}; rounds {rounds}; "
          f"the tuning round starts from {'the code defaults' if start_file == 'default' else 'data/calibration/' + start_file}; logs in {log_dir}")
    sys.stdout.flush()
    if "tune" in rounds:
        names = [f"{stem_tune}_{s}.json" for s in tune_seeds]
        files = [REPO / "data" / "mock" / nm for nm in names]
        if a.reuse and all(f.exists() for f in files) and reusable(files, tune_blocks):
            print("  tune: reusing the records " + ", ".join(names))
        else:
            files = run_workers("tune", tune_seeds, n_per, a.hours, a.prior, start_file, names, log_dir, stale_ok=start_file != "default", lag_h=a.lag_h)
        try:
            write_manifest(manifest_of("tune"), "tune", files, tune_blocks, extra={"state_noise_file": start_file, "hours": a.hours, "prior": a.prior, "workers": a.workers})
        except ValueError as e:
            raise SystemExit(f"the tuning round's records are not the round's: {e}") from e
        print(f"  wrote {manifest_of('tune')}")
        tune = load_round(files)
        print_round("the tuning round (before the identification)", tune, a.hours)
        rates, init, factors, fit = identify(tune, a.cap)
        print(format_fit(fit, UNITS, CHANNELS))
        print(format_lags(fit["lags"]))
        print("  the identification, each group's factor on its rate: " + ", ".join(f"{g} x{factors[g]:.2f}" for g in GROUPS))
        record = {"rates": rates, "init_sd": init, "fields": tune["fields"], "node_jitter": tune["node_jitter"], "events": tune["events"], "units": UNITS,
                  "physics": {"fingerprint": tune["physics"], "commit": git_commit(), "date": dt.date.today().isoformat(),
                              "what": "BrisketEstimator.physics_fingerprint: src/**/*.jl and the filter's own files; every driver refuses the file when the tree's differs, the tune round alone starts from stale rates"},
                  "method": "one autocovariance least-squares step on the tuning round (Odelson et al. 2006, fixed-gain time-varying form): the products of the "
                            "standardized innovations at every pair of updates inside the lag window, less the filter's own white model, regressed on the change of "
                            "each group's variance rate under the gains the filter used, pooled over the cooks, the rate capped; a group the innovations cannot see "
                            "keeps its numbers, the initial widths are kept; the fresh round is the check and the round the numbers are quoted from; "
                            "the bench's plant is the model's own physics, so the rates are a floor",
                  "bench": {"cooks_per_round": n_cooks, "cooks_per_worker": n_per, "workers": a.workers, "hours": a.hours, "prior": a.prior, "cap": a.cap, "lag_h": a.lag_h,
                            "tune": {"seeds": tune["seeds"], "skipped": tune["skipped"], "records": [f.name for f in files], "manifest": manifest_of("tune").name, "started_from": start_file,
                                     "physics": tune["physics"], "stale_ok": start_file != "default",
                                     "rates": tune["rates"], "init_sd": tune["init_sd"], "factors": factors, "als": fit,
                                     "nees_dof": {g: tune["groups"][g]["nees_dof"] for g in GROUPS},
                                     "nees_dof_cook_median": {g: tune["groups"][g]["cook_median"] for g in GROUPS}, "nees_dof_cook_max": {g: tune["groups"][g]["cook_max"] for g in GROUPS},
                                     "nis_dof": tune["nis_dof"], "latents_nees16": tune["latents_nees16"], "latents_per_cook": tune["latents_per_cook"], "gap": tune["gap"],
                                     "temps_nees3": tune["temps_nees3"], "temps_per_cook": tune["temps_per_cook"], "whiteness": tune["whiteness"]}}}
        rates_out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"  wrote {rates_out}")
        # the fit's figure: per channel and lag the sample autocovariance against the model before and after
        cmd = [sys.executable, str(REPO / "estimator" / "python" / "plot_als_fit.py"), str(manifest_of("tune")), tag, str(rates_out)]
        res = subprocess.run(cmd, cwd=REPO / "app", capture_output=True, text=True)
        (log_dir / "plot_als_fit.py.log").write_text(res.stdout + res.stderr, encoding="utf-8")
        print(res.stdout.strip())
        if res.returncode != 0:
            print(res.stderr.strip())
            raise SystemExit("plot_als_fit.py failed")
        sys.stdout.flush()
    if "fresh" in rounds:
        if a.variant and not rates_out.exists():
            rates_out = RATES_FILE                                                    # a variant's fresh round runs on the rates as earned
        if not rates_out.exists():
            raise SystemExit(f"no rates file at {rates_out}: run the tuning round first")
        sn_file = rates_out.name if rates_out == RATES_FILE else str(rates_out)      # the worker takes a name in data/calibration or an absolute path
        names = [f"{stem_fresh}_{s}.json" for s in fresh_seeds]
        files = [REPO / "data" / "mock" / nm for nm in names]
        record = json.loads(rates_out.read_text(encoding="utf-8"))
        expected_noise = {**record["rates"], **{g + "0": v for g, v in record["init_sd"].items()}, "fields": record["fields"], "node_jitter": record["node_jitter"], **record.get("events", {})}
        if a.reuse and all(f.exists() for f in files) and reusable(files, fresh_blocks, physics=record["physics"]["fingerprint"], state_noise=expected_noise):
            print("  fresh: reusing the records " + ", ".join(names))
        else:
            files = run_workers("fresh", fresh_seeds, n_per, a.hours, a.prior, sn_file, names, log_dir, lag_h=a.lag_h)   # strict: the rates must be this tree's
        try:                                                                          # the round's records, on the file's physics and rates, or nothing
            write_manifest(manifest_of("fresh"), "fresh", files, fresh_blocks, physics=record["physics"]["fingerprint"], state_noise=expected_noise,
                           extra={"state_noise_file": sn_file, "hours": a.hours, "prior": a.prior, "workers": a.workers})
        except ValueError as e:
            raise SystemExit(f"the fresh round's records are not the round's: {e}") from e
        print(f"  wrote {manifest_of('fresh')}")
        fresh = load_round(files)
        print_round("the fresh round (the rates as written, seeds never seen)", fresh, a.hours)
        if a.variant:                                                                 # a variant records its numbers beside the rates, never in them
            rates_out = REPO / "scratchpad" / f"state_noise_{a.variant}.json"
        record.setdefault("bench", {})["fresh"] = {"seeds": fresh["seeds"], "skipped": fresh["skipped"], "records": [f.name for f in files], "manifest": manifest_of("fresh").name, "physics": fresh["physics"],
                                                   "nees_dof": {g: fresh["groups"][g]["nees_dof"] for g in GROUPS},
                                                   "nees_dof_cook_median": {g: fresh["groups"][g]["cook_median"] for g in GROUPS}, "nees_dof_cook_max": {g: fresh["groups"][g]["cook_max"] for g in GROUPS},
                                                   "nis_dof": fresh["nis_dof"], "latents_nees16": fresh["latents_nees16"], "latents_per_cook": fresh["latents_per_cook"], "gap": fresh["gap"],
                                                   "temps_nees3": fresh["temps_nees3"], "temps_per_cook": fresh["temps_per_cook"], "whiteness": fresh["whiteness"]}
        rates_out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"  wrote {rates_out}")
        render(manifest_of("fresh"), tag, log_dir)
    print("done")


if __name__ == "__main__":
    main()
