"""Composition/boundary diagram for the writeup (criterion 1).

Run from app/:  uv run python -u ../scripts/render_diagram.py
Writes figures/composition_diagram.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.4)
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.12", linewidth=1.4)
WHITE = dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.95)


def box(x, y, w, h, fc, ec, title, lines, title_size=11, size=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, **BOX))
    ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.76 - 0.36 * i, ln, ha="center",
                va="center", fontsize=size)


def arrow(p, q, color, lw=1.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw))


def label(x, y, s, color, size=8.4):
    ax.text(x, y, s, ha="center", va="center", fontsize=size, color=color,
            bbox=WHITE)


# ---- JAX orchestrator (top) ----
box(1.4, 4.75, 10.2, 1.45, "#eef3fb", "#3b6ea5",
    "JAX orchestrator  (Python)",
    ["coupled_cook: K=2 Picard, unrolled  ·  jax.jit / jax.grad",
     "NumPyro MAP + NUTS   ·   Optax control   ·   E[J] over posterior"])

# ---- boundary (labels sit clear of boxes and line) ----
ax.plot([0.3, 12.7], [4.05, 4.05], ls=(0, (6, 4)), color="#888", lw=1.2)
# centered in the arrow-free span between the pit-side and meat-side crossings
ax.text(6.5, 4.05, "Tesseract boundary · apply / abstract_eval / VJP",
        ha="center", va="center", fontsize=8.8, color="#555", style="italic",
        bbox=WHITE)

# ---- pit tesseract (left) ----
box(0.6, 0.85, 4.6, 2.55, "#fdf1e7", "#c96a1b",
    "brisket-pit  (Julia)",
    ["lumped charcoal pit — 3 ODEs (Tsit5)",
     "energy · fuel burn · Minion ignition spread",
     "AD: exact dense ForwardDiff",
     "Jacobian, transposed to a VJP"])

# ---- meat tesseract (right) ----
box(7.8, 0.85, 4.6, 2.55, "#fdeaea", "#b03a3a",
    "brisket-meat  (Julia)",
    ["1D slab heat + moisture PDE + bark",
     "162 stiff ODEs (FBDF) · smoothed stall",
     "AD: SciMLSensitivity continuous adjoint",
     "(InterpolatingAdjoint + EnzymeVJP)"])

# FancyBboxPatch pads its VISIBLE border 0.12 beyond the nominal rect;
# connect arrows to the padded edge with a small clearance so heads/tails
# touch the border instead of piercing it.
PAD, CLR = 0.12, 0.03
JAX_BOT = 4.75 - PAD - CLR      # visible bottom edge of the JAX box
LOW_TOP = 3.40 + PAD + CLR      # visible top edge of the component boxes
PIT_R = 5.20 + PAD + CLR        # visible right edge of pit box
MEAT_L = 7.80 - PAD - CLR       # visible left edge of meat box

# ---- coupling arrows (in the gap between the boxes) ----
arrow((PIT_R, 2.85), (MEAT_L, 2.85), "#b03a3a")
label(6.5, 3.13, r"$T_{pit}(t),\ \dot m(t)$", "#b03a3a")
arrow((MEAT_L, 1.40), (PIT_R, 1.40), "#c96a1b")
label(6.5, 1.12, r"$q_{meat}(t)$", "#c96a1b")
label(6.5, 2.12, "two-way coupling\n(K = 2, unrolled)\nsmoke = burn rate",
      "#555", size=7.4)

# ---- boundary crossings: primal down (blue), outputs/cotangents up ----
arrow((2.0, JAX_BOT), (2.0, LOW_TOP), "#3b6ea5")
label(2.0, 4.44, r"$\varphi,\; u(t),\; q$", "#3b6ea5")
arrow((3.8, LOW_TOP), (3.8, JAX_BOT), "#c96a1b")
label(3.8, 3.72, r"$T_{pit},\, m_{fuel},\, \dot m$", "#c96a1b")
arrow((9.2, JAX_BOT), (9.2, LOW_TOP), "#3b6ea5")
label(9.2, 4.44, r"$\theta,\, T_{pit}(t),\, \dot m(t),\, t_{wrap}$", "#3b6ea5")
arrow((11.0, LOW_TOP), (11.0, JAX_BOT), "#b03a3a")
label(11.0, 3.72, r"$T_{core},\, j_{evap},\, q,\, C,\, w,\, L^{*}$", "#b03a3a")

# ---- gradient annotation (its own band at the bottom) ----
ax.text(6.5, 0.32,
        "one  jax.grad  of the composed log-density  =  4 VJP crossings of the "
        "boundary  (2 continuous-adjoint solves + 2 forward-mode transposes)",
        ha="center", fontsize=9.4, color="#333", fontweight="bold")

out = REPO / "figures" / "composition_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
