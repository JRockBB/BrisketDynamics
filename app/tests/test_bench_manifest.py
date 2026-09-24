"""The bench round's manifest (estimator/python/bench_manifest.py): records are accepted only when they carry the
round's seed blocks, one physics and one state noise, the expected ones when given; a record of another bench
(other seeds, other physics, other rates) is refused with the reason; a renderer resolves a manifest to its
validated records and a pattern to a glob."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "estimator" / "python"))
import bench_manifest as bm  # noqa: E402


def record(path: Path, seeds: list[int], physics: str = "abc123", t_pit: float = 2.5, skipped: list[int] | None = None) -> Path:
    d = {"cooks": [{"seed": s, "temps": [], "batches": []} for s in seeds], "skipped_seeds": skipped or [], "physics": physics,
         "state_noise": {"T": 0.5, "T_pit": t_pit, "fields": "offset", "node_jitter": 0.003}, "state_noise_file": "state_noise.json", "names": [], "prior_file": "p", "edges_h": []}
    path.write_text(json.dumps(d), encoding="utf-8")
    return path


def test_the_round_s_records_validate_and_others_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "MOCK", tmp_path)
    a = record(tmp_path / "coverage_filter_11.json", [11, 12, 13])
    b = record(tmp_path / "coverage_filter_14.json", [14, 15, 16])
    blocks = [[11, 12, 13], [14, 15, 16]]
    m = bm.validate_records([a, b], blocks, physics="abc123", state_noise={"T": 0.5, "T_pit": 2.5, "fields": "offset", "node_jitter": 0.003})
    assert m["seeds"] == [11, 12, 13, 14, 15, 16] and m["n_cooks"] == 6 and m["physics"] == "abc123"
    stale = record(tmp_path / "coverage_filter_16.json", [16, 17, 18], physics="old000", t_pit=1.0)
    with pytest.raises(ValueError, match="carries seeds"):
        bm.validate_records([a, stale], blocks)
    with pytest.raises(ValueError, match="different physics"):
        bm.validate_records([a, record(tmp_path / "x.json", [14, 15, 16], physics="old000")], blocks)
    with pytest.raises(ValueError, match="other state noise"):
        bm.validate_records([a, record(tmp_path / "y.json", [14, 15, 16], t_pit=1.0)], blocks)
    with pytest.raises(ValueError, match="expected"):
        bm.validate_records([a, b], blocks, physics="zzz")
    with pytest.raises(ValueError, match="records for"):
        bm.validate_records([a], blocks)
    # a skipped seed is allowed when the record says so
    c = record(tmp_path / "coverage_filter_21.json", [21, 23], skipped=[22])
    assert bm.validate_records([c], [[21, 22, 23]])["skipped"] == [22]


def test_the_manifest_round_trips_and_a_renderer_resolves_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bm, "MOCK", tmp_path)
    a = record(tmp_path / "coverage_filter_11.json", [11, 12])
    b = record(tmp_path / "coverage_filter_13.json", [13, 14])
    path = tmp_path / "bench_manifest_fresh.json"
    m = bm.write_manifest(path, "fresh", [a, b], [[11, 12], [13, 14]], physics="abc123", extra={"hours": 14.0})
    assert m["round"] == "fresh" and m["records"] == ["coverage_filter_11.json", "coverage_filter_13.json"] and m["hours"] == 14.0
    files, loaded = bm.resolve_inputs(str(path))
    assert [f.name for f in files] == m["records"] and loaded["seeds"] == [11, 12, 13, 14]
    assert "manifest" in capsys.readouterr().out
    files, none = bm.resolve_inputs("coverage_filter_1*.json")
    assert none is None and [f.name for f in files] == ["coverage_filter_11.json", "coverage_filter_13.json"]
    # a record changed after the manifest was written is refused at load
    record(tmp_path / "coverage_filter_13.json", [13, 14], t_pit=1.0)
    with pytest.raises(SystemExit, match="not the round's"):
        bm.load_manifest(path)
    b.unlink()
    with pytest.raises(SystemExit, match="not there"):
        bm.load_manifest(path)
