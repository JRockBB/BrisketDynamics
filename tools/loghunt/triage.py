#!/usr/bin/env python3
"""Score cached images for HeaterMeter-chart-ness and build a contact sheet.

The HM web UI (flot) has a stable fingerprint: dark plot background,
bluish gridlines, a saturated orange pit trace, a saturated green meat
trace, often a light-blue fan fill. Scores are heuristics for RANKING —
a human or the agent makes the pick from the contact sheet; nothing
downstream trusts these numbers.

Usage (from tools/loghunt/):
    python triage.py [--candidates candidates.json] [--min-score 0.5]

Writes triage.json (ranked) and contact_sheet.png.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def fingerprint(path: Path) -> dict:
    """HM-UI likeness in [0, 1] per cue, plus an overall score."""
    im = np.asarray(Image.open(path).convert("RGB"), dtype=int)
    h, w = im.shape[:2]
    if h < 200 or w < 400:
        return {"score": 0.0, "why": "too small"}
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    n = R.size
    dark = ((R < 90) & (G < 100) & (B < 110)).mean()
    grid = ((np.abs(R - 72) < 30) & (np.abs(G - 98) < 30) & (np.abs(B - 125) < 30)).mean()
    pit = ((R > 150) & (R - B > 60) & (R - G > 60) & (G > 60) & (G < 170)).sum() / n
    meat = ((G > 140) & (G - R > 40) & (G - B > 80)).sum() / n
    fan = ((B - R > 70) & (B > 130)).sum() / n
    # trace presence saturates quickly: a few thousand pixels is a full trace
    s_pit = min(pit * 400, 1.0)
    s_meat = min(meat * 400, 1.0)
    s_fan = min(fan * 150, 1.0)
    s_dark = min(dark * 2.5, 1.0)
    s_grid = min(grid * 30, 1.0)
    score = 0.30 * s_pit + 0.30 * s_meat + 0.15 * s_fan + 0.15 * s_dark + 0.10 * s_grid
    return {
        "score": round(float(score), 3),
        "pit": round(float(s_pit), 2),
        "meat": round(float(s_meat), 2),
        "fan": round(float(s_fan), 2),
        "dark": round(float(s_dark), 2),
        "grid": round(float(s_grid), 2),
        "w": w,
        "h": h,
    }


def contact_sheet(rows: list[dict], out: Path, n_cols: int = 4, thumb=(320, 180)):
    """Thumbnail grid of the ranked candidates for eyeball review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    if n == 0:
        return
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.4, n_rows * 2.2), squeeze=False
    )
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, r in zip(axes.ravel(), rows, strict=False):
        im = Image.open(HERE / "cache" / r["file"]).convert("RGB")
        im.thumbnail(thumb)
        ax.imshow(im)
        ax.set_title(
            f"{r['file']}  s={r['score']:.2f}\n{r.get('author', '?')}", fontsize=6
        )
    fig.suptitle("loghunt triage — ranked HeaterMeter-chart candidates", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    """Score cached images and write triage.json + contact_sheet.png."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.json")
    ap.add_argument("--min-score", type=float, default=0.35)
    args = ap.parse_args()
    cand = json.loads((HERE / args.candidates).read_text(encoding="utf-8"))
    by_file = {}
    for p in cand:
        for f in p.get("cached", []):
            by_file.setdefault(
                f, {"file": f, "author": p["author"], "post_url": p["post_url"]}
            )
    rows = []
    for f, meta in by_file.items():
        fp = fingerprint(HERE / "cache" / f)
        rows.append({**meta, **fp})
    rows.sort(key=lambda r: -r["score"])
    keep = [r for r in rows if r["score"] >= args.min_score]
    (HERE / "triage.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    contact_sheet(keep, HERE / "contact_sheet.png")
    print(
        f"{len(rows)} images scored; {len(keep)} >= {args.min_score}; "
        "wrote triage.json + contact_sheet.png"
    )
    for r in keep[:12]:
        print(f"  {r['score']:.2f}  {r['file']}  {r['author']}  {r['post_url']}")


if __name__ == "__main__":
    main()
