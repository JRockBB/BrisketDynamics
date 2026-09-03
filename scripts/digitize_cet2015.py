"""Digitize the experimental curves of Papasidero et al. (2015), CET 43:151.

Source: the open-access PDF at https://www.aidic.it/cet/15/43/026.pdf
(DOI 10.3303/CET1543026). Usage:

    uv run --with pypdfium2 python digitize_cet2015.py <path-to-026.pdf>

Renders page 5 at 4x, extracts Figure 3 (core / 5-mm-depth / intermediate
temperatures) and Figure 4 (dry-basis water content), and writes
data/validation/cet2015_extraction.json plus an overlay-check figure.
Axis anchors verified against tick-label centers (Fig 4's bottom detected
line is the 0 gridline at page row 2077; asserted below).

Office-palette masks: blue (79,129,189) core, red (192,80,77) 5-mm-depth
probe, green (155,187,89) intermediate; gray 191/217 gridlines. Solid =
paper model (continuous per column), dashed = experiment (gappy) — per
column the cluster nearest the tracked solid path is the model, the rest
is experiment; gaps interpolate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SP = str(REPO / "data" / "validation")
pdf = pdfium.PdfDocument(sys.argv[1])
im = np.asarray(pdf[4].render(scale=4.0).to_pil().convert("RGB")).astype(int)
R, G, B = im[..., 0], im[..., 1], im[..., 2]

def near(v, t, tol=28):
    return np.abs(v - t) < tol

MASKS = {
    "blue": near(R, 79) & near(G, 129) & near(B, 189),
    "red": near(R, 192) & near(G, 80) & near(B, 77),
    "green": near(R, 155) & near(G, 187) & near(B, 89),
}
gray = near(R, 217, 20) & near(G, 217, 20) & near(B, 217, 20) & (np.abs(R - G) < 6) & (np.abs(G - B) < 6)
gray2 = near(R, 191, 12) & near(G, 191, 12) & near(B, 191, 12) & (np.abs(R - G) < 6) & (np.abs(G - B) < 6)

def gridlines(r0, r1, c0, c1):
    sub = (gray | gray2)[r0:r1, c0:c1]
    rows = np.where(sub.mean(axis=1) > 0.55)[0] + r0
    cols = np.where(sub.mean(axis=0) > 0.55)[0] + c0
    def grp(a):
        return [float(np.mean(g)) for g in np.split(a, np.where(np.diff(a) > 3)[0] + 1)] if len(a) else []
    return grp(rows), grp(cols)

def clusters(mask, r0, r1, c):
    rr = np.where(mask[r0:r1, c])[0]
    if not len(rr):
        return []
    return [float(np.mean(g)) + r0 for g in np.split(rr, np.where(np.diff(rr) > 5)[0] + 1)]

def track(mask, r0, r1, c0, c1):
    """Per column: continuous (solid) path + leftover (dashed) clusters."""
    solid, dashed = {}, {}
    last = None
    for c in range(c0, c1):
        cl = clusters(mask, r0, r1, c)
        if not cl:
            continue
        if last is None:
            # first colored column: solid starts here (dashed may too;
            # take the cluster count as-is, nearest-to-none = first)
            last = cl[0]
        d = [abs(x - last) for x in cl]
        i = int(np.argmin(d))
        solid[c] = cl[i]
        last = cl[i]
        rest = [x for j, x in enumerate(cl) if j != i]
        if rest:
            dashed[c] = rest[int(np.argmin([abs(x - last) for x in rest]))]
    return solid, dashed

# ---- Figure 3 ----------------------------------------------------------------
gr, gc = gridlines(360, 1040, 800, 1490)
print("fig3 grid rows:", [round(x, 1) for x in gr])
print("fig3 grid cols:", [round(x, 1) for x in gc])
# y: top gridline = 100 C ... bottom = 0 C; x: leftmost = 0 min, 20-min steps
y100, y0 = gr[0], gr[-1]
x0 = gc[0]
# the last detected column is the frame RIGHT edge, not a 20-min line
px_per_min = float(np.median(np.diff(gc[:-1]))) / 20.0
row2C = lambda r: (y0 - r) / (y0 - y100) * 100.0
col2min = lambda c: (c - x0) / px_per_min
print(f"fig3: {px_per_min:.3f} px/min, y0 {y0}, y100 {y100}")

out = {"fig3": {}, "fig4": {}}
for name, key in (("core", "blue"), ("depth5mm", "red"), ("intermediate", "green")):
    m = MASKS[key] & ~np.zeros_like(MASKS[key])
    solid, dashed = track(MASKS[key], int(y100) - 8, int(y0) + 4, int(x0), int(gc[-1]) + 1)
    for kind, d in (("model", solid), ("exp", dashed)):
        cc = np.array(sorted(d))
        tt = np.array([col2min(c) for c in cc])
        vv = np.array([row2C(d[c]) for c in cc])
        # resample to 2-min grid over the common span
        tg = np.arange(0.0, tt.max() // 2 * 2 + 0.1, 2.0)
        vg = np.interp(tg, tt, vv)
        out["fig3"][f"{name}_{kind}"] = {"t_min": tg.tolist(), "T_C": [round(v, 2) for v in vg]}
        print(f"{name}_{kind}: {len(cc)} cols, t {tt.min():.1f}-{tt.max():.1f}, T {vv.min():.1f}-{vv.max():.1f}")

# ---- Figure 4 ----------------------------------------------------------------
gr4, gc4 = gridlines(1450, 2120, 800, 1620)
print("fig4 grid rows:", [round(x, 1) for x in gr4])
print("fig4 grid cols:", [round(x, 1) for x in gc4])
# first detected row is the frame top (not a 0.5-gridline); anchor on the
# clean lower lines: bottom = 0 (confirmed against the y-label centers),
# uniform 0.5-steps
y0_4 = gr4[-1]
step4 = float(np.median(np.diff(gr4[1:])))
assert abs(y0_4 - 2077) < 3, y0_4
x0_4 = gc4[0]
px_per_min4 = float(np.median(np.diff(gc4[:-1]))) / 20.0
row2W = lambda r: (y0_4 - r) / step4 * 0.5
col2min4 = lambda c: (c - x0_4) / px_per_min4
y35 = y0_4 - 7 * step4

# red squares = experiment points
mred = MASKS["red"]
sub = mred[int(y35) - 10:int(y0_4), int(x0_4) - 20:int(gc4[-1]) + 6]
rr, cc = np.where(sub)
pts = {}
for r, c in zip(rr, cc):
    kx = c // 12
    pts.setdefault(kx, []).append((r, c))
sq = []
for k, v in sorted(pts.items()):
    a = np.array(v)
    if len(a) > 40:  # a filled square, not a stray
        sq.append((col2min4(a[:, 1].mean() + x0_4 - 20), row2W(a[:, 0].mean() + y35 - 10)))
# merge adjacent bins of the same square
merged = []
for t, w in sq:
    if merged and abs(merged[-1][0] - t) < 4:
        merged[-1] = ((merged[-1][0] + t) / 2, (merged[-1][1] + w) / 2)
    else:
        merged.append((t, w))
print("fig4 experiment squares (t_min, W_db):", [(round(t, 1), round(w, 3)) for t, w in merged])
out["fig4"]["experiment_points"] = [{"t_min": round(t, 1), "W_db": round(w, 3)} for t, w in merged]

# blue model line
solid, dashed = track(MASKS["blue"], int(y35) - 10, int(y0_4), int(x0_4), int(gc4[-1]) + 1)
cc = np.array(sorted(solid))
tt = np.array([col2min4(c) for c in cc])
vv = np.array([row2W(solid[c]) for c in cc])
tg = np.arange(0.0, tt.max() // 2 * 2 + 0.1, 2.0)
out["fig4"]["model"] = {"t_min": tg.tolist(), "W_db": [round(v, 3) for v in np.interp(tg, tt, vv)]}
print(f"fig4 model: W {vv.max():.2f} -> {vv.min():.2f}")

with open(SP + r"\cet2015_extraction.json", "w") as f:
    json.dump(out, f)
print("wrote data/validation/cet2015_extraction.json")


# ---- overlay check -----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].imshow(im[350:1050, 780:1500])
for name, color in (("core", "tab:cyan"), ("depth5mm", "magenta"), ("intermediate", "yellow")):
    d = out["fig3"][name + "_exp"]
    cc = [x0 + t * px_per_min - 780 for t in d["t_min"]]
    rr = [y0 - v / 100.0 * (y0 - y100) - 350 for v in d["T_C"]]
    axes[0].plot(cc, rr, "--", color=color, lw=1.2, label=name + " exp (extracted)")
axes[0].legend(fontsize=8); axes[0].set_title("Figure 3: extracted experiment over original")
axes[1].imshow(im[1420:2100, 780:1660])
d = out["fig4"]["model"]
cc = [x0_4 + t * px_per_min4 - 780 for t in d["t_min"]]
rr = [y0_4 - v / 0.5 * step4 - 1420 for v in d["W_db"]]
axes[1].plot(cc, rr, "--", color="magenta", lw=1.2, label="model (extracted)")
for p_ in out["fig4"]["experiment_points"]:
    axes[1].plot(x0_4 + p_["t_min"] * px_per_min4 - 780, y0_4 - p_["W_db"] / 0.5 * step4 - 1420,
                 "x", color="lime", ms=12, mew=2)
axes[1].legend(fontsize=8); axes[1].set_title("Figure 4: extracted over original (x = exp points)")
for a in axes: a.axis("off")
fig.tight_layout()
fig.savefig(SP + r"\cet2015_overlay_check.png", dpi=110)
print("wrote data/validation/cet2015_overlay_check.png")
