#!/usr/bin/env python3
"""Digitize the Mache WSM-18.5 brisket cook from a HeaterMeter web-UI screenshot.

Source: https://tvwbb.com/threads/post-your-heatermeter-graphs.42779/post-474182
See data/real/PROVENANCE.md for cook metadata, timeline, and uncertainty estimates.

Usage:
    python digitize_cook_log.py <screenshot.png> [outdir]

Outputs (in outdir, default '.'):
    mache_brisket_cook.csv   uniform 2-min grid, t=0 at meat-on lid opening
    overlay_check.png        extracted CSV re-projected onto the source chart

Method: per-pixel-column color-mask extraction; axis calibration from detected
gridlines, verified against the known 225 F setpoint line (extracts 225.16 F).
Gaps > 6 min are left NaN, never bridged. Deterministic (no RNG).

Requires: numpy, pandas, pillow, matplotlib
"""
import sys
import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------------------------------------------------------
# Calibration constants (derived from gridline detection; see detect_gridlines)
# All pixel coordinates are in the cropped main-chart frame.
CROP = (slice(280, 640), slice(440, 1150))  # (rows, cols) of main chart in screenshot
ROW_TOP, ROW_BOT = 10.4, 336.2       # pixel rows of 260 F / 40 F (also 100% / 0% fan)
PX_PER_20F = 29.618                  # vertical px per 20 F (gridline spacing)
COL_0030 = 95.0                      # pixel col of the 00:30 gridline
PX_PER_HR = 30.1                     # horizontal px per hour
MEAT_ON_OFFSET_MIN = 15.9            # meat-on lid opening precedes 00:30 by this
COOK_END_MIN = 1141.0                # meat pulled (relative to meat-on)
GRID_STEP_MIN = 2.0
MAX_GAP_MIN = 6.0                    # interpolate across gaps up to this; else NaN
SETPOINT_F = 225.0                   # constant by config; occluded by pit trace

def row2F(r):    return 260.0 - (r - ROW_TOP) * 20.0 / PX_PER_20F
def F2row(F):    return ROW_TOP + (260.0 - F) * PX_PER_20F / 20.0
def row2pct(r):  return (ROW_BOT - r) / (ROW_BOT - ROW_TOP) * 100.0
def pct2row(p):  return ROW_BOT - (p / 100.0) * (ROW_BOT - ROW_TOP)
def col2min(c):  return (c - COL_0030) / PX_PER_HR * 60.0 + MEAT_ON_OFFSET_MIN
def min2col(t):  return COL_0030 + (t - MEAT_ON_OFFSET_MIN) / 60.0 * PX_PER_HR


def detect_gridlines(im):
    """Verification helper: recover gridline rows/cols from which the constants
    above were derived. Not used in the extraction path, but run it if you swap
    in a different screenshot of the same UI."""
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    gridish = (np.abs(R - 72) < 25) & (np.abs(G - 98) < 25) & (np.abs(B - 125) < 25)
    H, W = gridish.shape

    def group(idx):
        out, cur = [], [idx[0]]
        for v in idx[1:]:
            (cur.append(v) if v - cur[-1] <= 2 else (out.append(np.mean(cur)), cur := [v]))
        out.append(np.mean(cur))
        return out

    rows = [i for i in range(H) if gridish[i].sum() > W * 0.5]
    cols = [j for j in range(W) if gridish[:, j].sum() > H * 0.5]
    return (group(rows) if rows else []), (group(cols) if cols else [])


def extract(im):
    """Per-column extraction of all channels. Returns a raw (per-pixel-column) frame."""
    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    rt, rb = int(ROW_TOP) + 3, int(ROW_BOT) - 2   # stay inside the plot frame

    line_masks = {
        # pit temp: saturated orange (incl. anti-aliased variants)
        'pit':  (R > 150) & (R - B > 60) & (R - G > 60) & (G > 60) & (G < 170),
        # meat temp: saturated green
        'meat': (G > 140) & (G - R > 40) & (G - B > 80),
    }
    # fan duty: light-blue fill/trace; strictly bluer than the bluish gridlines
    fan_mask = (B - R > 70) & (B > 130)
    # ambient: thin low-saturation gray line; extracted below with continuity tracking
    amb_mask = (np.abs(R - G) < 25) & (np.abs(G - B) < 32) & (R > 58) & (R < 185)
    # lid-event bands (white / pale yellow vertical streaks)
    white = (R > 180) & (G > 180) & (B > 160)
    yellow = (R > 200) & (G > 170) & (B < 170) & (R - B > 60)

    cols = np.arange(30, 668)
    recs, prev_amb = [], None
    for c in cols:
        rec = {'col': c, 't_raw': col2min(c)}
        for name, m in line_masks.items():
            rr = np.where(m[rt:rb, c])[0]
            rec[name] = row2F(np.median(rr) + rt) if len(rr) else np.nan
        rr = np.where(fan_mask[rt:rb, c])[0]
        rec['fan_pct'] = row2pct(rr.min() + rt) if len(rr) else np.nan   # top edge of fill

        # ambient: pick the line-like run (1-8 px) nearest the previous column's row,
        # rejecting jumps > 12 px, so frame pixels / text / the fan fill can't hijack it
        rr = np.where(amb_mask[159:rb, c])[0] + 159   # ambient never exceeds ~160 F
        runs, cur = [], None
        for v in rr:
            if cur and v - cur[-1] <= 2: cur.append(v)
            else:
                if cur: runs.append(cur)
                cur = [v]
        if cur: runs.append(cur)
        runs = [r for r in runs if 1 <= len(r) <= 8]
        if runs:
            pick = (min(runs, key=lambda r: abs(np.median(r) - prev_amb))
                    if prev_amb is not None else max(runs, key=len))
            med = np.median(pick)
            if prev_amb is None or abs(med - prev_amb) <= 12:
                prev_amb = med
                rec['amb'] = row2F(med)
            else:
                prev_amb = None
        else:
            prev_amb = None
        rec['band'] = bool(white[rt:rb, c].sum() > 60 or yellow[rt:rb, c].sum() > 60)
        recs.append(rec)
    return pd.DataFrame(recs)


def resample(raw):
    """Trim to the cook window, apply timeline (t=0 at meat-on), resample to a
    uniform grid; gaps > MAX_GAP_MIN become NaN instead of fabricated chords."""
    cook = raw[(raw.t_raw >= 0) & (raw.t_raw <= COOK_END_MIN)].copy()
    cook.loc[cook.t_raw < 12, 'meat'] = np.nan          # probe settling after load-in

    grid = np.arange(0.0, COOK_END_MIN, GRID_STEP_MIN)
    out = pd.DataFrame({'t_min': grid})

    def interp_gap(t, v):
        vi = np.interp(grid, t, v)
        idx = np.searchsorted(t, grid)
        left = t[np.clip(idx - 1, 0, len(t) - 1)]
        right = t[np.clip(idx, 0, len(t) - 1)]
        vi[np.minimum(np.abs(grid - left), np.abs(right - grid)) > MAX_GAP_MIN] = np.nan
        return vi

    for src, dst in [('pit', 'pit_F'), ('meat', 'meat_F'), ('amb', 'amb_F'),
                     ('fan_pct', 'fan_pct')]:
        s = cook.dropna(subset=[src])
        out[dst] = interp_gap(s.t_raw.values, s[src].values)
    out['setpoint_F'] = SETPOINT_F
    out['lid_event'] = ((out.t_min > 1086) & (out.t_min < 1100)) | (out.t_min < 14)
    return out.round(1)


def overlay(im_pil, df, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14.2, 7.2), dpi=100)
    ax.imshow(im_pil)
    c = min2col(df.t_min)
    ax.plot(c, F2row(df.pit_F), 'w--', lw=1, alpha=.85, label='pit')
    ax.plot(c, F2row(df.meat_F), 'k--', lw=1, alpha=.85, label='meat')
    ax.plot(c, F2row(df.amb_F), 'y--', lw=.8, alpha=.8, label='ambient')
    ax.plot(c, pct2row(df.fan_pct), 'm--', lw=.7, alpha=.7, label='fan %')
    ax.legend(loc='upper center', ncol=4, fontsize=9, framealpha=.9)
    ax.set_axis_off()
    ax.set_title('Digitization check: dashed = extracted CSV over original', fontsize=11)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
    full = np.array(Image.open(sys.argv[1]).convert('RGB'))
    chart = full[CROP]

    raw = extract(chart)
    df = resample(raw)

    # Built-in calibration check against the known setpoint line
    R, G, B = chart[..., 0].astype(int), chart[..., 1].astype(int), chart[..., 2].astype(int)
    sp = (R > 140) & (G < 70) & (B < 70)
    vals = [row2F(np.median(np.where(sp[13:334, c])[0]) + 13)
            for c in range(30, 668) if sp[13:334, c].any()]
    sp_err = abs(np.mean(vals) - SETPOINT_F)
    print(f'setpoint calibration: extracted {np.mean(vals):.2f} F '
          f'(true 225.00, err {sp_err:.2f} F)')
    assert sp_err < 0.5, 'calibration drift — check CROP and gridline constants'

    df.to_csv(f'{outdir}/mache_brisket_cook.csv', index=False)
    overlay(Image.fromarray(chart), df, f'{outdir}/overlay_check.png')
    print(f'{len(df)} rows -> {outdir}/mache_brisket_cook.csv; overlay written')


if __name__ == '__main__':
    main()
