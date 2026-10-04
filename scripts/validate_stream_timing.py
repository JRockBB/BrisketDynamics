"""Audit saved causal benchmark traces, including a clearly labelled prefix.

Usage: python scripts/validate_stream_timing.py records.json [audit.json]
This checks scheduling and accounting, not statistical calibration.
"""
import json
import math
from pathlib import Path
import sys


def audit(data):
    pace = data["pace"]
    if not math.isfinite(pace) or pace <= 0:
        raise ValueError("A causal processing-backlog trace with positive pace is required")
    runs = [(c, "complete") for c in data.get("cooks", [])]
    runs += [(c, "failed") for c in data.get("failures", [])]
    if data.get("active_cook"):
        runs.append((data["active_cook"], "incomplete"))
    results = []
    for cook, status in runs:
        tape = cook.get("input_records", [])
        rows = cook.get("temps", [])
        errors = []

        def check(condition, message):
            if not condition:
                errors.append(message)

        def close(a, b):
            return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6)

        check(bool(tape), "missing input tape")
        check(all(a["t_s"] <= b["t_s"] for a, b in zip(tape, tape[1:])), "input tape is not sorted")
        consumed = 0
        available = 0.0
        for number, row in enumerate(rows, 1):
            label = f"update {number}: "
            arrivals = row["arrivals"]
            n = len(arrivals)
            expected = tape[consumed:consumed + n]
            check(n > 0 and n == row["n_readings"], label + "reading count mismatch")
            check(arrivals == [{"sensor": a["sensor"], "t_s": a["t_s"]} for a in expected],
                  label + "dropped, duplicated, reordered, or changed input arrival")
            if not expected or not arrivals:
                continue
            start, finish = row["processing_start_s"], row["processing_finish_s"]
            check(close(start, max(available, expected[0]["t_s"])), label + "idle waiting or clock rewind")
            check(all(a["t_s"] <= start for a in arrivals), label + "future observation in update")
            check(close(finish, start + pace * row["wall_s"]), label + "incorrect compute duration")
            check(close(row["t_h"] * 3600, arrivals[-1]["t_s"]), label + "incorrect state timestamp")
            check(close(row["oldest_reading_latency_s"], finish - arrivals[0]["t_s"]), label + "incorrect oldest latency")
            check(close(row["newest_reading_latency_s"], finish - arrivals[-1]["t_s"]), label + "incorrect newest latency")
            check(row["n_nodes"] <= data["lag_max"] + 1, label + "window exceeds node limit")
            consumed += n
            check(consumed == len(tape) or tape[consumed]["t_s"] > start,
                  label + "already-arrived observation left out of backlog")
            available = finish
        check(consumed <= len(tape), "more consumed readings than input readings")
        if status == "complete":
            check(consumed == len(tape), "completed run has unconsumed input readings")
            check(all(r["committed"] and r["converged"] for r in rows), "completed run has an uncommitted or unconverged update")
        for edge in cook.get("batches", []):
            ready = [r for r in rows if r["committed"] and r["processing_finish_s"] <= edge["t_h"] * 3600]
            check(bool(ready) and close(edge["state_t_h"], ready[-1]["t_h"]), "score uses an unavailable estimate")
        results.append({
            "seed": cook["seed"], "status": status, "trace_checks_pass": not errors,
            "completed_horizon_pass": status == "complete" and not errors,
            "updates": len(rows), "input_readings": len(tape), "consumed_readings": consumed,
            "uncommitted_updates": sum(not r["committed"] for r in rows),
            "updates_with_multiple_readings": sum(r["n_readings"] > 1 for r in rows),
            "updates_with_queued_readings": sum(r["processing_start_s"] > r["arrivals"][0]["t_s"] + 1e-6 for r in rows),
            "nodes_added": sum(r["nodes_added"] for r in rows),
            "last_state_h": rows[-1]["t_h"] if rows else None,
            "next_input_h": tape[consumed]["t_s"] / 3600 if consumed < len(tape) else None,
            "last_available_h": available / 3600,
            "max_oldest_latency_s": max((r["oldest_reading_latency_s"] for r in rows), default=None),
            "errors": errors,
        })
    return {"backend": data.get("backend_fingerprint"), "update_policy": data.get("update_policy"),
            "scope": "Causal scheduling and input accounting only; incomplete runs check a prefix.",
            "runs": results, "trace_checks_pass": bool(results) and all(r["trace_checks_pass"] for r in results)}


if __name__ == "__main__":
    source = Path(sys.argv[1])
    result = audit(json.loads(source.read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    print(text, end="")
    sys.exit(0 if result["trace_checks_pass"] else 1)
