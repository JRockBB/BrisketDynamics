"""Demo weather timing and recovery of completed runs (no Julia required)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "run_mpc", Path(__file__).resolve().parents[2] / "scripts/run_mpc.py"
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_front_is_two_hours_after_planned_load():
    assert demo.planned_disturbance_time({"best_t_load_h": 2.35}) == pytest.approx(
        4.35 * 3600
    )
    assert demo.planned_disturbance_time({"t_load_h": 1.5}) == pytest.approx(3.5 * 3600)
    assert demo.planned_disturbance_time({}) == 2 * 3600
    with pytest.raises(ValueError, match="meat-on time"):
        demo.planned_disturbance_time({"best_T_load": 366.48})


def test_completed_cook_checkpoint_roundtrip_and_invalidation(tmp_path):
    calls = []

    def run():
        calls.append(1)
        return {
            **{key: np.array([0.0, 1.0]) for key in demo.TRAJ_KEYS},
            "doneness": np.float64(0.9),
            "final_state": object(),
        }

    path = tmp_path / "nominal.json"
    demo.cached_cook(path, "same-inputs", run)
    restored = demo.cached_cook(path, "same-inputs", run)
    assert len(calls) == 1
    assert restored["doneness"] == 0.9
    np.testing.assert_array_equal(restored["t"] / 3600, [0.0, 1 / 3600])
    demo.cached_cook(path, "changed-inputs", run)
    assert len(calls) == 2
