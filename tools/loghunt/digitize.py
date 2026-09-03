#!/usr/bin/env python3
"""Digitize a HeaterMeter web-UI cook chart from a per-log TOML description.

Generalization of digitize_mache_reference.py (the frozen method behind
data/cook_log/mache_brisket_cook.csv): the flot palette and extraction
method are constant across HM screenshots; what varies per log — axis
anchor values, the meat-on moment, cook end, lid events, setpoint — is
the judgment layer a human or the agent records in ``logs/<slug>/log.toml``
after reading the post. Everything mechanical (chart cropping, px-per-
degree calibration, trace extraction, resampling, gap handling) is
derived from gridline detection.

Usage (from tools/loghunt/):
    python digitize.py logs/<slug>/log.toml [--compare some.csv]

Outputs into logs/<slug>/: <slug>.csv, overlay_check.png, PROVENANCE_STUB.md.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import tomllib
from PIL import Image

HERE = Path(__file__).resolve().parent


# --- palettes: the HM web UI ships in a few color themes ----------------------
# hm_classic: steel-blue plot, bluish gridlines, one green meat trace, gray
#   ambient (the 2013 Mache render). hm_dark: black plot, gray gridlines, two
#   meat traces (light-green "point", teal "flat"), no ambient trace.


def _classic_masks(R, G, B):
    return {
        "grid": (np.abs(R - 72) < 25) & (np.abs(G - 98) < 25) & (np.abs(B - 125) < 25),
        "pit": (R > 150) & (R - B > 60) & (R - G > 60) & (G > 60) & (G < 170),
        "meat": {"meat": (G > 140) & (G - R > 40) & (G - B > 80)},
        "fan": (B - R > 70) & (B > 130),
        "amb": (np.abs(R - G) < 25) & (np.abs(G - B) < 32) & (R > 58) & (R < 185),
        "setpoint": (R > 140) & (G < 70) & (B < 70),
    }


def _dark_masks(R, G, B):
    return {
        "grid": (np.abs(R - G) < 14) & (np.abs(G - B) < 14) & (R > 52) & (R < 95),
        "pit": (R > 150) & (R - B > 60) & (R - G > 60) & (G > 60) & (G < 170),
        "meat": {
            "point": (G > 150) & (G - R > 60) & (G - B > 110),
            "flat": (G > 110) & (G - R > 60) & (G - B > 20) & (G - B < 75) & (B > 60),
        },
        "fan": (B > G) & (B - R > 55) & (B > 85),
        # gray ambient line when present (harmless when absent: no hits -> NaN)
        "amb": (np.abs(R - G) < 20) & (np.abs(G - B) < 20) & (R > 100) & (R < 200),
        "setpoint": (R > 120) & (G < 60) & (B < 60),
    }


def _flot_jpeg_masks(R, G, B):
    """The 2013 flot render served through a lossy pipeline (JPEG artifacts
    and a semi-transparent watermark that lightens everything it covers):
    the hues survive, so the masks are looser hue tests than hm_classic,
    tuned to still exclude the dark-blue ground, the light-neutral
    watermark, and the washed-out tan lid bands."""
    return {
        "grid": (np.abs(R - 72) < 25) & (np.abs(G - 98) < 25) & (np.abs(B - 125) < 25),
        "pit": (R > 140) & (R - B > 45) & (R - G > 25) & (G < 210),
        "meat": {"meat": (G > 95) & (G - R > 22) & (G - B > 12)},
        "fan": (B - R > 35) & (B > 110) & (B - G > 10),
        "amb": None,
        "setpoint": (R > 110) & (R - G > 55) & (R - B > 70) & (G < 80),
    }


PALETTES = {
    "hm_classic": _classic_masks,
    "hm_dark": _dark_masks,
    "hm_flot_jpeg": _flot_jpeg_masks,
}
_PALETTE = _classic_masks  # set per log in main()


# --- gridline detection (from the reference's verification helper) -----------


def gridline_mask(im: np.ndarray) -> np.ndarray:
    """Palette gridline color mask."""
    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    return _PALETTE(R, G, B)["grid"]


def _group(idx: list[int]) -> list[float]:
    out, cur = [], [idx[0]]
    for v in idx[1:]:
        if v - cur[-1] <= 2:
            cur.append(v)
        else:
            out.append(float(np.mean(cur)))
            cur = [v]
    out.append(float(np.mean(cur)))
    return out


def detect_gridlines(im: np.ndarray, frac: float = 0.35):
    """Return grouped gridline row/col positions (fraction-of-span vote per line)."""
    g = gridline_mask(im)
    H, W = g.shape
    rows = [i for i in range(H) if g[i].sum() > W * frac]
    cols = [j for j in range(W) if g[:, j].sum() > H * frac]
    return (_group(rows) if rows else []), (_group(cols) if cols else [])


def auto_crop(full: np.ndarray) -> tuple[slice, slice]:
    """Bounding box of the dense gridline region (the plot frame)."""
    g = gridline_mask(full)
    # a plot row/col contains many gridline-colored pixels even off the lines
    row_any = g.sum(axis=1)
    col_any = g.sum(axis=0)
    rr = np.where(row_any > 0.25 * np.max(row_any))[0]
    cc = np.where(col_any > 0.25 * np.max(col_any))[0]
    if rr.size == 0 or cc.size == 0:
        raise SystemExit("auto-crop failed: no gridline-like region; supply [extract].crop")
    return slice(int(rr[0]) - 4, int(rr[-1]) + 5), slice(int(cc[0]) - 4, int(cc[-1]) + 5)


# --- extraction (ported from the reference, parametrized) --------------------


def extract(im, row_top, row_bot, col_lo, col_hi, row2F, row2pct, col2min, amb_max_row, meat_probe, want_amb=True):
    """Per-pixel-column color-mask extraction of pit/meat probes/ambient/fan."""
    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    rt, rb = int(row_top) + 3, int(row_bot) - 2
    pal = _PALETTE(R, G, B)
    if not want_amb:
        pal["amb"] = None  # e.g. the gray trace is a dome probe, not ambient
    line_masks = {"pit": pal["pit"], "meat": pal["meat"][meat_probe]}
    for name, m in pal["meat"].items():
        if name != meat_probe:
            line_masks[f"meat2_{name}"] = m
    fan_mask = pal["fan"]
    amb_mask = pal["amb"]

    cols = np.arange(col_lo, col_hi)
    n = len(cols)
    out = {"t_raw": np.array([col2min(c) for c in cols])}
    for name in line_masks:
        out[name] = np.full(n, np.nan)
    out["amb"] = np.full(n, np.nan)
    out["fan_pct"] = np.full(n, np.nan)
    prev_amb = None
    for k, c in enumerate(cols):
        for name, m in line_masks.items():
            rr = np.where(m[rt:rb, c])[0]
            if len(rr):
                out[name][k] = row2F(np.median(rr) + rt)
        rr = np.where(fan_mask[rt:rb, c])[0]
        if len(rr):
            out["fan_pct"][k] = row2pct(rr.min() + rt)
        if amb_mask is None:
            continue
        rr = np.where(amb_mask[amb_max_row:rb, c])[0] + amb_max_row
        runs, cur = [], None
        for v in rr:
            if cur and v - cur[-1] <= 2:
                cur.append(v)
            else:
                if cur:
                    runs.append(cur)
                cur = [v]
        if cur:
            runs.append(cur)
        runs = [r for r in runs if 1 <= len(r) <= 8]
        if runs:
            pick = (
                min(runs, key=lambda r: abs(np.median(r) - prev_amb))
                if prev_amb is not None
                else max(runs, key=len)
            )
            med = np.median(pick)
            if prev_amb is None or abs(med - prev_amb) <= 12:
                prev_amb = med
                out["amb"][k] = row2F(med)
            else:
                prev_amb = None
        else:
            prev_amb = None
    return out


def resample(raw, cook_end_min, settle_min, lid_events, setpoint_f, step_min, max_gap):
    """Uniform-grid resample; gaps beyond max_gap become NaN, never chords."""
    keep = (raw["t_raw"] >= 0) & (raw["t_raw"] <= cook_end_min)
    t_raw = raw["t_raw"][keep]
    meat = raw["meat"][keep].copy()
    meat[t_raw < settle_min] = np.nan
    series = {
        "pit_F": raw["pit"][keep],
        "meat_F": meat,
        "amb_F": raw["amb"][keep],
        "fan_pct": raw["fan_pct"][keep],
    }
    for k in raw:
        if k.startswith("meat2_"):
            series[k.replace("meat2_", "") + "_F"] = raw[k][keep]
    grid = np.arange(0.0, cook_end_min, step_min)
    out = {"t_min": grid}

    def interp_gap(t, v):
        vi = np.interp(grid, t, v)
        idx = np.searchsorted(t, grid)
        left = t[np.clip(idx - 1, 0, len(t) - 1)]
        right = t[np.clip(idx, 0, len(t) - 1)]
        vi[np.minimum(np.abs(grid - left), np.abs(right - grid)) > max_gap] = np.nan
        return vi

    for dst, v in series.items():
        ok = np.isfinite(v)
        out[dst] = (
            np.round(interp_gap(t_raw[ok], v[ok]), 1)
            if ok.any()
            else np.full(len(grid), np.nan)
        )
    if setpoint_f is not None:
        out["setpoint_F"] = np.full(len(grid), float(setpoint_f))
    flag = grid < max(settle_min + 2, 14)
    for a, b in lid_events:
        flag |= (grid > a) & (grid < b)
    out["lid_event"] = flag
    return out


def write_csv(out: dict, path):
    """Write the column dict as CSV (blank cells for NaN)."""
    keys = list(out)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(out["t_min"])):
            row = []
            for k in keys:
                v = out[k][i]
                if k == "lid_event":
                    row.append(bool(v))
                elif isinstance(v, (float, np.floating)) and np.isnan(v):
                    row.append("")
                else:
                    row.append(round(float(v), 1))
            w.writerow(row)


def overlay(im_pil, df, F2row, pct2row, min2col, path):
    """Re-project the extracted CSV onto the source chart (the eyeball check)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14.2, 7.2), dpi=100)
    ax.imshow(im_pil)
    c = min2col(df["t_min"])
    ax.plot(c, F2row(df["pit_F"]), "w--", lw=1, alpha=0.85, label="pit")
    ax.plot(c, F2row(df["meat_F"]), "k--", lw=1, alpha=0.85, label="meat")
    if np.isfinite(df["amb_F"]).any():
        ax.plot(c, F2row(df["amb_F"]), "y--", lw=0.8, alpha=0.8, label="ambient")
    ax.plot(c, pct2row(df["fan_pct"]), "m--", lw=0.7, alpha=0.7, label="fan %")
    ax.legend(loc="upper center", ncol=4, fontsize=9, framealpha=0.9)
    ax.set_axis_off()
    ax.set_title("Digitization check: dashed = extracted CSV over original", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    """Digitize one log described by its log.toml."""
    ap = argparse.ArgumentParser()
    ap.add_argument("log_toml")
    ap.add_argument("--compare", default=None, help="existing CSV to diff against")
    args = ap.parse_args()
    toml_path = Path(args.log_toml)
    cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    outdir = toml_path.parent
    slug = cfg.get("slug", outdir.name)

    full = np.array(Image.open(HERE / cfg["source"]["image"]).convert("RGB"))
    ex = cfg.get("extract", {})
    if "crop" in ex:
        r0, r1, c0, c1 = ex["crop"]
        crop = (slice(r0, r1), slice(c0, c1))
    else:
        crop = auto_crop(full)
        print(f"auto-crop rows {crop[0].start}:{crop[0].stop} cols {crop[1].start}:{crop[1].stop}")
    chart = full[crop]

    global _PALETTE
    ex_palette = cfg.get("extract", {}).get("palette", "hm_classic")
    _PALETTE = PALETTES[ex_palette]
    ax_cfg = cfg["axes"]
    have_overrides = "px_per_step_v" in ax_cfg and "px_per_step_h" in ax_cfg and "x_anchor_col" in ax_cfg
    rows_g, cols_g = detect_gridlines(chart)
    if (len(rows_g) < 3 or len(cols_g) < 3) and not have_overrides:
        raise SystemExit(f"gridline detection too sparse (rows {len(rows_g)}, cols {len(cols_g)})")
    cal = cfg.get("calibration", {})
    y_top, y_bot = float(ax_cfg["y_top_f"]), float(ax_cfg["y_bottom_f"])
    step_f = float(ax_cfg.get("grid_step_f", 20.0))
    px_step_v = float(
        ax_cfg["px_per_step_v"] if "px_per_step_v" in ax_cfg else np.median(np.diff(rows_g))
    )

    # Vertical anchor, in order of trust: (1) the red setpoint line — an
    # ABSOLUTE temperature reference (gridlines only supply the px/step
    # scale; which gridline is which label is exactly what detection cannot
    # know when the top lines are occluded); (2) the topmost detected
    # gridline == y_top_f, with a loud warning.
    Rc = chart[..., 0].astype(int)
    Gc = chart[..., 1].astype(int)
    Bc = chart[..., 2].astype(int)
    row_sp = None
    if cal.get("setpoint_f") is not None:
        sp_mask = _PALETTE(Rc, Gc, Bc)["setpoint"]
        sp_rows = [
            float(np.median(np.where(sp_mask[:, c])[0]))
            for c in range(chart.shape[1])
            if sp_mask[:, c].any()
        ]
        if len(sp_rows) > 30:
            row_sp = float(np.median(sp_rows))
    if row_sp is not None:
        f_anchor, row_anchor = float(cal["setpoint_f"]), row_sp
        print(f"vertical anchor: setpoint line at row {row_sp:.1f} = {f_anchor} F")
    elif "row_top" in ax_cfg:
        f_anchor, row_anchor = y_top, float(ax_cfg["row_top"])
        print(f"vertical anchor: configured frame top row {row_anchor} = {y_top} F")
    else:
        f_anchor, row_anchor = y_top, rows_g[0]
        print(
            "WARNING: no setpoint line to anchor on; assuming the topmost "
            f"detected gridline is {y_top} F — verify with the overlay",
            file=sys.stderr,
        )

    def row2F(r):
        return f_anchor + (row_anchor - r) * step_f / px_step_v

    def F2row(F):
        return row_anchor - (F - f_anchor) * px_step_v / step_f

    # the plot frame rows follow from the anchored mapping
    row_top, row_bot = F2row(y_top), F2row(y_bot)
    n_expect = round((y_top - y_bot) / step_f)
    span_steps = (rows_g[-1] - rows_g[0]) / px_step_v
    if span_steps > n_expect + 0.35:
        print(
            f"WARNING: detected gridline span {span_steps:.2f} steps exceeds "
            f"the {n_expect} steps of [{y_bot}, {y_top}] F — check the axes block",
            file=sys.stderr,
        )

    def row2pct(r):
        return (row_bot - r) / (row_bot - row_top) * 100.0

    def pct2row(p):
        return row_bot - (p / 100.0) * (row_bot - row_top)

    x_step_min = float(ax_cfg.get("x_step_min", 30.0))
    px_step_h = float(ax_cfg.get("px_per_step_h", np.median(np.diff(cols_g))))
    # horizontal anchor: an explicit (column, clock-minute) pair when given
    # (recorded per log against the CACHED canonical image), else the
    # leftmost detected gridline at t_left_min
    if "x_anchor_col" in ax_cfg:
        col_anchor = float(ax_cfg["x_anchor_col"])
        min_anchor = float(ax_cfg["x_anchor_clock_min"])
    else:
        col_anchor = cols_g[0]
        min_anchor = float(ax_cfg["t_left_min"])
    meat_on = float(cfg["timeline"]["meat_on_min"])

    def col2min(c):  # minutes since meat-on
        return (c - col_anchor) / px_step_h * x_step_min + min_anchor - meat_on

    def min2col(t):
        return col_anchor + (t + meat_on - min_anchor) / x_step_min * px_step_h

    print(
        f"calibration: {px_step_v:.2f} px / {step_f:.0f} F, "
        f"{px_step_h / x_step_min * 60:.1f} px / hr, "
        f"frame rows {row_top:.0f}-{row_bot:.0f}"
    )

    tl = cfg["timeline"]
    amb_max_row = max(int(F2row(float(ex.get("amb_max_f", 160.0)))), int(row_top) + 3)
    col_lo = int(ex.get("col_lo", 2))
    col_hi = int(ex.get("col_hi", chart.shape[1] - 2))
    raw = extract(
        chart, row_top, row_bot, col_lo, col_hi, row2F, row2pct, col2min, amb_max_row,
        ex.get("meat_probe", "meat" if ex_palette == "hm_classic" else "flat"),
        want_amb=ex.get("amb", True),
    )
    df = resample(
        raw,
        float(tl["cook_end_min"]),
        float(tl.get("settle_min", 12.0)),
        [tuple(w) for w in tl.get("lid_events", [])],
        cal.get("setpoint_f"),
        float(ex.get("grid_step_min", 2.0)),
        float(ex.get("max_gap_min", 6.0)),
    )

    if row_sp is not None:
        spread = float(np.std(sp_rows)) * step_f / px_step_v
        print(
            f"setpoint line: {len(sp_rows)} columns, row spread "
            f"{spread:.2f} F-equivalent (anchor quality)"
        )

    csv_path = outdir / f"{slug}.csv"
    write_csv(df, csv_path)
    overlay(Image.fromarray(chart), df, F2row, pct2row, min2col, outdir / "overlay_check.png")
    n_rows = len(df["t_min"])
    print(f"{n_rows} rows -> {csv_path}; overlay written")

    stub = outdir / "PROVENANCE_STUB.md"
    if not stub.exists():
        s = cfg["source"]
        stub.write_text(
            f"# Provenance: {slug}.csv (STUB — finish before adopting into data/)\n\n"
            f"## Source\n- **Post:** {s.get('author', '?')}, {s['url']}\n"
            f"- **Image:** {s['image']} (cached locally, not committed), "
            f"retrieved {s.get('retrieved', '?')}. Forum content (c) its author; "
            "small factual data extraction with attribution.\n\n"
            "## Cook metadata (from post text)\n- TODO: cooker, fuel, vents, "
            "controller, meat cut/weight, wrap moments, weather.\n\n"
            f"## Timeline (t = 0 at meat-on)\n- meat-on at chart minute "
            f"{tl['meat_on_min']}; cook end {tl['cook_end_min']} min; lid events "
            f"{tl.get('lid_events', [])}.\n\n"
            "## Digitization\n- tools/loghunt/digitize.py with this directory's "
            "log.toml; gridline auto-calibration "
            f"({px_step_v:.2f} px/{step_f:.0f}F, {px_step_h / x_step_min * 60:.1f} px/hr); "
            "gaps > 6 min NaN, never bridged.\n\n"
            "## Estimated uncertainty\n- TODO: line width / calibration in F, "
            "time px, fan caveats (see data/cook_log/PROVENANCE.md for the template).\n",
            encoding="utf-8",
        )
        print(f"wrote {stub}")

    if args.compare:
        ref = np.genfromtxt(args.compare, delimiter=",", names=True)
        n = min(len(ref), len(df["t_min"]))
        print(f"compare vs {args.compare} ({n} rows):")
        for col in ("pit_F", "meat_F", "fan_pct"):
            if col in ref.dtype.names and col in df:
                a = np.asarray(df[col][:n], dtype=float)
                b = np.asarray(ref[col][:n], dtype=float)
                ok = np.isfinite(a) & np.isfinite(b)
                nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
                print(
                    f"  {col:8s} max|d|={np.max(np.abs(a[ok] - b[ok])):.2f} "
                    f"mean|d|={np.mean(np.abs(a[ok] - b[ok])):.3f} "
                    f"(n={int(ok.sum())}, nan-mismatch={nan_mismatch})"
                )


if __name__ == "__main__":
    main()
