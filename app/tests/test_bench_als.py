"""The bench's autocovariance least-squares step (estimator/python/als_fit.py) on synthetic sufficient statistics:
a planted change of the identified groups' variance rates is recovered within its standard error, a group the
rows carry no information on keeps its numbers, the cap binds, the record is JSON-strict, and cooks that ran on
different rates cannot be pooled."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "estimator" / "python"))
from als_fit import als_fit, format_fit, format_lags, lag_table, pool

GROUPS = ["T", "W", "C", "F", "S", "bark", "T_pit", "m_fuel", "m_pan", "T_pan", "T_fb", "ignited"]


def synthetic(rng: np.random.Generator, theta: np.ndarray, delta: np.ndarray, n_rows: int, active: list[int], scale: float = 0.05,
              noise: tuple[float, float] = (1.0, 1.0)) -> dict:
    """One cook's summary from rows whose coefficients are random on the active groups and zero elsewhere, in two
    blocks (the pit's squares at one update, its products across updates) with the noise sds given."""
    n_g = len(theta)
    half = n_rows // 2
    blocks, pairs, G, h, rss = [], [], np.zeros((n_g, n_g)), np.zeros(n_g), 0.0
    for same, sd, bn, base in ((True, noise[0], 0, 1.0), (False, noise[1], 1, 0.0)):
        B = np.zeros((half, n_g)); B[:, active] = rng.normal(size=(half, len(active))) * scale
        s = B @ delta + sd * rng.normal(size=half)
        prod = s + base
        blocks.append({"a": 1, "b": 1, "same": same, "G": (B.T @ B).tolist(), "h": (B.T @ s).tolist(), "rss": float(s @ s), "n": half,
                       "s_sum": float(s.sum()), "s_sq": float(s @ s)})
        pairs.append({"a": 1, "b": 1, "bin": bn, "n": half, "sum_product": float(prod.sum()), "sum_product2": float(prod @ prod),
                      "sum_base": float(base * half), "sum_b": B.sum(axis=0).tolist()})
        G += B.T @ B; h += B.T @ s; rss += float(s @ s)
    return {"groups": GROUPS, "theta": theta.tolist(), "lag_s": 7200.0, "bin_s": 120.0, "n_updates": 10, "n_rows": 2 * half, "rss": rss,
            "G": G.tolist(), "h": h.tolist(), "pairs": pairs, "blocks": blocks}


def test_fit_recovers_the_planted_change_and_holds_the_unseen():
    rng = np.random.default_rng(3)
    theta = np.array([0.25, 1e-4, 1e-5, 1e-5, 1e-8, 1e-4, 4.0, 1.0, 4e-4, 4.0, 4.0, 4e-4])
    delta = np.zeros(12); delta[6] = 8.0; delta[0] = -0.125                       # the pit's rate by √3, the field's by 1/√2, the fuel's unchanged
    active = [0, 6, 7]
    cooks = [synthetic(rng, theta, delta, 20000, active, noise=(3.0, 1.0)) for _ in range(3)]   # the squares three times as scattered
    pooled = pool(cooks)
    assert pooled["n_cooks"] == 3 and pooled["n_rows"] == 60000 and len(pooled["blocks"]) == 2
    fit = als_fit(pooled, cap=4.0)
    w = {(b["a"], b["b"], b["same"]): b["weight"] for b in fit["blocks"]}
    assert 7 < w[(1, 1, False)] / w[(1, 1, True)] < 11                             # the weighting found the variance ratio of nine
    for g in active:
        assert fit["identified"][g], GROUPS[g]
        se_theta = 2 * fit["se_rate"][g] * fit["rates_new"][g]
        assert abs(fit["theta_new"][g] - theta[g] - delta[g]) < 3 * se_theta, GROUPS[g]
    for g in range(12):
        if g in active:
            continue
        assert not fit["identified"][g] and fit["held"][GROUPS[g]] == "unidentified"
        assert fit["rates_new"][g] == pytest.approx(np.sqrt(theta[g])) and fit["factors"][g] == 1.0
    assert fit["rss_after"] < fit["rss_before"] and 0.8 < fit["sigma2"] < 1.2
    rows = lag_table(pooled, fit, {1: "pit probe"}, [0, 1, 2])
    assert [r["bin"] for r in rows] == [0, 1] and rows[0]["model_before"] == pytest.approx(1.0) and rows[1]["model_before"] == 0.0
    assert 2.5 < rows[0]["se"] / rows[1]["se"] < 3.5                                # the error bars from the products' own spread
    json.dumps(fit, allow_nan=False)                                               # the rates file is parsed strictly on the Julia side
    text = format_fit(fit, {g: "K" for g in GROUPS}, {1: "pit probe"}) + format_lags(rows)
    assert "identified" in text and "unidentified" in text
    # the leave-one-cook-out jackknife: on cooks that are draws of one process its standard errors agree with the
    # model's within a factor, the identified groups are identified in every leave-one-out fit, the record is JSON
    fit = als_fit(pooled, cap=4.0, summaries=cooks)
    jk = fit["jackknife"]
    assert jk["n_cooks"] == 3 and all(jk["identified_in"][g] == 0 for g in range(12) if g not in active)
    assert jk["identified_in"][6] == 3 and jk["identified_in"][7] == 3                # the pit and the fuel pinned by every two of the cooks
    assert jk["identified_in"][0] < 3                                                 # the field, borderline on three cooks, is not pinned by every two: what the jackknife is for
    for g in (6, 7):
        assert 0.1 < jk["se_rate"][g] / fit["se_rate"][g] < 10
    json.dumps(fit, allow_nan=False)
    assert "jackknife se" in format_fit(fit, {g: "K" for g in GROUPS}, {1: "pit probe"})


def test_the_cap_binds_and_a_small_rate_the_innovations_want_larger_moves():
    rng = np.random.default_rng(5)
    theta = np.full(12, 1.0)
    delta = np.zeros(12); delta[6] = 400.0                                        # far beyond the cap of four on the rate
    fit = als_fit(pool([synthetic(rng, theta, delta, 20000, [6], scale=0.02)]), cap=4.0)
    assert fit["identified"][6] and fit["cap_bound"][6] and fit["rates_new"][6] == pytest.approx(4.0)
    assert fit["factors"][6] == pytest.approx(4.0) and fit["moved_sigmas"][6] > 2.0
    # a rate small today whose column alone cannot pin it to within half, but which the fit moves by many standard
    # errors: it enters and moves to the cap; the same column without a planted move is held
    theta = np.full(12, 1.0); theta[7] = 2.5e-3
    delta = np.zeros(12); delta[7] = 1.0                                          # about seven standard errors of the fit
    fit = als_fit(pool([synthetic(rng, theta, delta, 20000, [7], scale=0.05)]), cap=4.0)
    assert fit["identified"][7] and fit["cap_bound"][7] and fit["rates_new"][7] == pytest.approx(4 * np.sqrt(2.5e-3))
    null = synthetic(rng, theta, np.zeros(12), 20000, [7], scale=0.05)
    null["h"] = [0.0] * 12                                                        # a sample whose rows are uncorrelated with the column: no move at all
    for b in null["blocks"]:
        b["h"] = [0.0] * 12
    fit = als_fit(pool([null]), cap=4.0)
    assert not fit["identified"][7] and fit["moved_sigmas"][7] == 0.0 and fit["rates_new"][7] == pytest.approx(np.sqrt(2.5e-3))


def test_cooks_on_different_rates_do_not_pool():
    rng = np.random.default_rng(7)
    a = synthetic(rng, np.full(12, 1.0), np.zeros(12), 100, [6])
    b = synthetic(rng, np.full(12, 2.0), np.zeros(12), 100, [6])
    with pytest.raises(ValueError):
        pool([a, b])
    with pytest.raises(ValueError):
        pool([])
