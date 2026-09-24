"""The manifest of one bench round: which records are the round's, and the checks a renderer makes before it
pools them.

A bench round's workers each write one record (coverage_filter.jl), named by the block's first seed. Names
carry the seed block, and the block changes with the worker count, so a glob on the round's stem can pick up a
record of an earlier bench beside the round's own (it did on 2026-09-23: two records of 2026-09-17, on other
physics and other rates, were pooled into the consistency and coverage figures). The driver therefore writes a
manifest per round, data/mock/bench_manifest_<round><tag>.json, listing the round's records with their seeds,
the cook count, the physics fingerprint and the state noise they ran with; every renderer takes the manifest,
reads exactly its records, and refuses them when their seeds, count, physics or rates are not the manifest's.

The renderers still take a glob for the older records that have no manifest (the ridge, the batch and the
smoother comparisons); a manifest is recognised by its name.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MOCK = REPO / "data" / "mock"


def record_summary(path: Path) -> dict:
    """What a record says about itself: its seeds, count, physics and state noise."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"name": Path(path).name, "seeds": [int(c["seed"]) for c in d["cooks"]], "n_cooks": len(d["cooks"]), "physics": d.get("physics"),
            "state_noise": d.get("state_noise"), "state_noise_file": d.get("state_noise_file"), "skipped": d.get("skipped_seeds", [])}


def _same_noise(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a is b
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not np.isclose(float(x), float(y), rtol=1e-9, atol=0.0):
                return False
        elif x != y:
            return False
    return True


def validate_records(files: list[Path], blocks: list[list[int]], physics: str | None = None, state_noise: dict | None = None) -> dict:
    """The records against the round's blocks (one list of seeds per record, in order): each record carries exactly
    its block's seeds (a skipped seed is allowed when the record says it skipped it), every record the same physics
    and the same state noise, the physics and the noise the expected ones when given. Returns the round's seeds,
    physics and state noise. Raises ValueError naming the first mismatch."""
    if len(files) != len(blocks):
        raise ValueError(f"{len(files)} records for {len(blocks)} seed blocks")
    summaries = [record_summary(f) for f in files]
    for s, block in zip(summaries, blocks):
        expected = [x for x in block if x not in s["skipped"]]
        if s["seeds"] != expected:
            raise ValueError(f"{s['name']} carries seeds {s['seeds']}, the round's block is {block} (skipped {s['skipped']})")
    phys = {s["physics"] for s in summaries}
    if len(phys) != 1:
        raise ValueError(f"the records ran on {len(phys)} different physics: {sorted(str(p) for p in phys)}")
    if physics is not None and summaries[0]["physics"] != physics:
        raise ValueError(f"the records ran on physics {summaries[0]['physics']}, expected {physics}")
    for s in summaries[1:]:
        if not _same_noise(s["state_noise"], summaries[0]["state_noise"]):
            raise ValueError(f"{s['name']} ran on other state noise than {summaries[0]['name']}")
    if state_noise is not None and not _same_noise(summaries[0]["state_noise"], state_noise):
        raise ValueError(f"the records ran on other state noise than expected ({summaries[0]['name']})")
    seeds = sorted(x for s in summaries for x in s["seeds"])
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"a seed appears twice across the records: {seeds}")
    return {"seeds": seeds, "n_cooks": len(seeds), "physics": summaries[0]["physics"], "state_noise": summaries[0]["state_noise"],
            "records": [s["name"] for s in summaries], "blocks": blocks, "skipped": sorted(x for s in summaries for x in s["skipped"])}


def write_manifest(path: Path, round_name: str, files: list[Path], blocks: list[list[int]], physics: str | None = None,
                   state_noise: dict | None = None, extra: dict | None = None) -> dict:
    """Validate the round's records and write its manifest."""
    m = validate_records(files, blocks, physics=physics, state_noise=state_noise)
    m.update({"round": round_name, "date": dt.date.today().isoformat(), **(extra or {})})
    Path(path).write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def load_manifest(path: Path) -> tuple[list[Path], dict]:
    """The manifest's records, validated against it (seeds, count, physics, state noise) before any renderer pools them."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    files = [MOCK / name for name in m["records"]]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"the manifest {Path(path).name} names records that are not there: {missing}")
    try:
        chk = validate_records(files, m["blocks"], physics=m.get("physics"), state_noise=m.get("state_noise"))
    except ValueError as e:
        raise SystemExit(f"the records of {Path(path).name} are not the round's: {e}") from e
    if chk["seeds"] != m["seeds"] or chk["n_cooks"] != m["n_cooks"]:
        raise SystemExit(f"the records of {Path(path).name} carry seeds {chk['seeds']}, the manifest says {m['seeds']}")
    return files, m


def is_manifest(arg: str) -> bool:
    return Path(arg).name.startswith("bench_manifest_") and arg.endswith(".json")


def resolve_inputs(arg: str) -> tuple[list[Path], dict | None]:
    """A renderer's first argument: a round's manifest (the records it names, validated) or a glob on data/mock."""
    if is_manifest(arg):
        p = Path(arg)
        if not p.is_absolute():
            p = MOCK / p.name
        files, m = load_manifest(p)
        print(f"the round's manifest {p.name}: {m['n_cooks']} cooks (seeds {m['seeds']}) in {len(files)} records, physics {m['physics']}")
        return files, m
    return [Path(f) for f in sorted(glob.glob(str(MOCK / arg)))], None
