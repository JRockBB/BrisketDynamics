from dataclasses import replace

import numpy as np
import pytest
from brisket_app import mpc
from brisket_app.pilot import Pilot, PilotConfig
from brisket_app.pit_io import PitStatus


def status(t, *, tpit=350.0, tcore=320.0, fan=40.0):
    return PitStatus(t, 375.0, tpit, tcore, fan, fan, 0.0)


class FakePit:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.setpoints = []
        self.events = []

    def read(self):
        return next(self.statuses)

    def set_setpoint(self, value):
        self.setpoints.append(value)

    def execute_wrap(self, t_now):
        self.events.append(("wrap", t_now))

    def execute_pull(self, t_now):
        self.events.append(("pull", t_now))

    def execute_add(self, t_now):
        self.events.append(("add", t_now))

    def execute_spritz(self, t_now, **kwargs):
        self.events.append(("spritz", t_now, kwargs))


def make_pilot(pit, **cfg):
    return Pilot(
        pit,
        np.ones(8),
        np.array([1.0e4, 20.0, 1.0e-4, 1.0e7, 0.2, 1.0e-4, 1.0]),
        PilotConfig(**cfg),
        log=lambda *_: None,
    )


def test_hardware_clock_is_normalized_and_must_be_monotone():
    origin = 1_700_000_000.0
    pilot = make_pilot(FakePit([status(origin)]))
    assert pilot._elapsed_status(status(origin + 60.0)).t == 60.0
    with pytest.raises(RuntimeError, match="moved backward"):
        pilot._elapsed_status(status(origin + 59.0))
    with pytest.raises(RuntimeError, match="fan duty"):
        pilot._elapsed_status(status(origin + 61.0, fan=float("nan")))


def test_unacknowledged_event_stays_pending():
    pilot = make_pilot(FakePit([status(10.0)]), auto_events=False)
    pilot._advise("wrap", 120.0)
    assert not pilot.wrapped
    assert pilot.t_wrap_exec is None
    assert pilot.pending_events == {"wrap"}

    pilot.acknowledge_event("wrap", 125.0, source="test")
    assert pilot.wrapped
    assert pilot.t_wrap_exec == 125.0
    assert not pilot.pending_events


def test_backend_executes_all_modeled_events():
    pit = FakePit([status(0.0)])
    pilot = make_pilot(pit)
    pilot.dt_spritz = 900.0
    pilot.spritz_dose = 0.08

    for name in ("wrap", "pull", "add", "spritz"):
        pilot._advise(name, 300.0)

    assert pilot.wrapped and pilot.pulled and pilot.added and pilot.spritzing
    assert [event[0] for event in pit.events] == ["wrap", "pull", "add", "spritz"]
    assert pit.events[-1][2] == {"dt_spritz": 900.0, "spritz_dose": 0.08}


def test_shadow_mode_sends_no_setpoint_or_event_command():
    pit = FakePit([status(1000.0), status(1060.0)])
    pilot = make_pilot(pit, shadow_mode=True)
    pilot.t_wrap = 30.0

    meas = pilot.execute_segment(60.0, np.array([0.0, 60.0]), np.array([370.0, 380.0]))

    assert meas["t"] == [60.0]
    assert pit.setpoints == []
    assert pit.events == []
    assert not pilot.wrapped
    assert pilot.pending_events == {"wrap"}


def test_prediction_and_assimilation_forward_add_and_spritz(monkeypatch):
    pilot = make_pilot(FakePit([status(1000.0)]))
    pilot.t_wrap = 100.0
    pilot.t_pull = 200.0
    pilot.t_add = 300.0
    pilot.t_spritz = 400.0
    pilot.dt_spritz = 900.0
    pilot.spritz_dose = 0.08
    calls = []

    def advance(state, _u, dt, *_args, **kwargs):
        calls.append(kwargs)
        return replace(state, t=state.t + dt), {"tpit": np.array([350.0, 351.0])}

    monkeypatch.setattr(mpc, "advance_plant", advance)
    grids = type("Grids", (), {"u_tknots": np.array([0.0, 60.0])})()
    pilot.predict_segment(np.array([0.4, 0.5]), grids, 60.0)
    assert calls[-1]["t_add_seg"] == 300.0
    assert calls[-1]["t_spritz_seg"] == 400.0

    pilot.acknowledge_event("add", 20.0, source="test")
    pilot.acknowledge_event("spritz", 30.0, source="test")
    meas = {"u": [0.4, 0.5], "tpit": [350.0, 351.0]}
    pilot.assimilate(meas, 60.0, 351.0)
    assert calls[-1]["t_add_seg"] == 20.0
    assert calls[-1]["t_spritz_seg"] == 30.0
    assert calls[-1]["dt_spritz_seg"] == 900.0
    assert calls[-1]["spritz_dose_seg"] == 0.08


def test_advance_plant_carries_complete_handoff(monkeypatch):
    state = mpc.CookState.fresh(
        mpan0=1.2,
        tpan0=295.0,
        tfb0=430.0,
        wrap_latch0=0.4,
        c_duct=60.0,
    )
    captured = {}

    def coupled(*_args, **kwargs):
        captured.update(kwargs["ics"])
        field = np.arange(3.0)
        return {
            "tpit": np.array([350.0, 351.0]),
            "tcore": np.array([320.0, 321.0]),
            "tsurf": np.array([330.0, 331.0]),
            "m_fuel": np.array([6.0, 5.9]),
            "ignited": np.array([0.8, 0.81]),
            "m_pan": np.array([1.2, 1.1]),
            "t_pan": np.array([295.0, 296.0]),
            "t_firebox": np.array([430.0, 425.0]),
            "wrap_latch": np.array([0.4, 0.7]),
            "t_field_final": field,
            "w_field_final": field,
            "c_field_final": field,
            "f_field_final": field,
            "salt_field_final": field,
            "bark_b": np.array([0.1, 0.2]),
            "bark_s": np.array([0.2, 0.3]),
            "ring": np.asarray(0.004),
        }

    monkeypatch.setattr(mpc, "coupled_cook", coupled)
    monkeypatch.setattr(mpc.jnp, "asarray", np.asarray)
    new, _ = mpc.advance_plant(
        state, np.array([0.4, 0.5]), 60.0, np.ones(8), np.ones(7), 300.0
    )

    assert captured["m_pan0"] == 1.2
    assert captured["t_firebox0"] == 430.0
    assert captured["wrap_latch0"] == 0.4
    assert captured["c_duct"] == 60.0
    assert new.mpan0 == 1.1
    assert new.tpan0 == 296.0
    assert new.tfb0 == 425.0
    assert new.wrap_latch0 == 0.7
