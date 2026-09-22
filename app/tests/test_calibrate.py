"""Real-cook AR(1) likelihood: segments, ρ=0 ≡ iid, constant bias is cheaper at high ρ."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from brisket_app.calibrate import ar1_log_prob, ar1_observe, contiguous_runs
from brisket_app.cooklog import load
from numpyro.infer.util import log_density


def test_contiguous_runs_splits_on_holes():
    mask = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    assert contiguous_runs(mask) == ((2, 3, 4), (6, 7))
    assert contiguous_runs(np.zeros(4, dtype=bool)) == ()
    assert contiguous_runs(np.ones(3, dtype=bool)) == ((0, 1, 2),)
    assert contiguous_runs(np.array([1, 0, 1], dtype=bool)) == ((0,), (2,))


def test_cooklog_lid_breaks_the_chain():
    log = load(dt_min=10.0)
    pit = contiguous_runs(log["mask_pit"])
    core = contiguous_runs(log["mask_core"])
    assert len(pit) >= 2
    assert len(core) >= 2
    assert sum(len(r) for r in pit) == int(log["mask_pit"].sum())
    assert sum(len(r) for r in core) == int(log["mask_core"].sum())
    # a gap between first two pit runs is a masked (lid) sample
    assert pit[0][-1] + 1 < pit[1][0]


def test_ar1_rho0_matches_iid():
    y = jnp.array([1.0, 1.4, 0.7, 1.1])
    mu = jnp.array([1.0, 1.0, 1.0, 1.0])
    sigma = 0.5
    got = ar1_log_prob(y, mu, sigma, 0.0)
    want = dist.Normal(mu, sigma).log_prob(y).sum()
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-10)


def test_ar1_constant_offset_cheaper_at_high_rho():
    mu = jnp.zeros(20)
    y = mu + 2.0
    sigma = 1.0
    iid = ar1_log_prob(y, mu, sigma, 0.0)
    ar = ar1_log_prob(y, mu, sigma, 0.95)
    assert np.asarray(ar) > np.asarray(iid)


def test_ar1_observe_matches_log_prob():
    y = jnp.array([0.2, 0.5, 0.1])
    mu = jnp.array([0.0, 0.1, 0.0])
    sigma, rho = 0.4, 0.7

    def model():
        ar1_observe("seg", y, mu, sigma, rho)

    lp_model, _ = log_density(model, (), {}, {})
    lp_direct = ar1_log_prob(y, mu, sigma, rho)
    np.testing.assert_allclose(np.asarray(lp_model), np.asarray(lp_direct), rtol=1e-10)


def test_as_cooklog_aligns_damper_with_observation_grid():
    from brisket_app import synth
    from brisket_app.coupled import HR, CookGrids

    grids = CookGrids(t_end=6 * HR, dt_exchange=600.0, n_u=len(synth.U_COOK))
    nt = len(grids.tknots)
    log = synth.as_cooklog(
        {
            "t": grids.tknots,
            "u_knots": synth.U_COOK,
            "obs_pit": np.full(nt, 350.0),
            "obs_core": np.full(nt, 290.0),
            "grids": grids,
        }
    )
    assert log["u"].shape == log["t"].shape
    np.testing.assert_allclose(log["u"][0], synth.U_COOK[0])
    np.testing.assert_allclose(log["u"][-1], synth.U_COOK[-1])


def test_cooklog_pit_gap_is_interpolated_but_never_observed(tmp_path):
    """A span of missing pit samples is filled on the grid (the damper and the starts need a value) but masked
    out of the likelihood, as a missing meat span already was."""
    from brisket_app import cooklog

    rows = ["t_min,pit_F,meat_F,amb_F,fan_pct,setpoint_F,lid_event"]
    for k in range(37):  # 0..360 min at 10 min
        t = 10 * k
        pit = "" if 100 <= t <= 150 else f"{225 + 0.01 * t:.2f}"
        rows.append(f"{t},{pit},{60 + 0.2 * t:.2f},70,30,225,false")
    csv_path = tmp_path / "gap.csv"
    csv_path.write_text("\n".join(rows) + "\n")
    cooklog.LOGS["gap_test"] = {"csv": csv_path, "t_end_min": None, "t_amb_K": 300.0, "start_at_core_K": None}
    try:
        log = cooklog.load(dt_min=10.0, log="gap_test")
    finally:
        del cooklog.LOGS["gap_test"]
    t_min = log["t"] / 60.0
    in_gap = (t_min >= 100) & (t_min <= 150)
    assert np.all(np.isfinite(log["obs_pit"]))            # interpolated for the grid
    assert not log["mask_pit"][in_gap].any()              # never observed
    assert log["mask_pit"][~in_gap].all()                 # everything else is
    assert log["mask_core"].all()
