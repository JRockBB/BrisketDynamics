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

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.6)
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.12", linewidth=1.4)


def box(x, y, w, h, fc, ec, title, lines, title_size=10.5, size=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, **BOX))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - 0.34 * i, ln, ha="center",
                va="center", fontsize=size)


def arrow(p, q, color, label=None, lx=0, ly=0, style="-|>", lw=1.8, ls="-",
          connstyle="arc3,rad=0.0", size=13):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=size,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=connstyle))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label,
                ha="center", va="center", fontsize=8.2, color=color)


# ---- JAX orchestrator (top) ----
box(1.2, 4.0, 8.6, 1.35, "#eef3fb", "#3b6ea5",
    "JAX orchestrator  (Python)",
    ["coupled_cook: K=2 Picard, unrolled  ·  jax.jit / jax.grad",
     "NumPyro MAP + NUTS   ·   Optax control   ·   E[J] over posterior"])

# ---- boundary ----
ax.plot([0.4, 10.6], [3.55, 3.55], ls=(0, (6, 4)), color="#888", lw=1.2)
ax.text(10.55, 3.7, "Tesseract boundary — apply · abstract_eval · VJP",
        ha="right", fontsize=8.6, color="#555", style="italic")

# ---- pit tesseract (left) ----
box(0.7, 0.6, 4.3, 2.3, "#fdf1e7", "#c96a1b",
    "brisket-pit  (Julia)",
    ["lumped charcoal pit — 2 ODEs (Tsit5)",
     "energy balance · fuel burn + smooth taper",
     "AD: exact dense ForwardDiff Jacobian,",
     "transposed to a VJP"])

# ---- meat tesseract (right) ----
box(6.0, 0.6, 4.3, 2.3, "#fdeaea", "#b03a3a",
    "brisket-meat  (Julia)",
    ["1D slab heat + moisture PDE — 120 stiff",
     "ODEs (FBDF) · smoothed evaporative stall",
     "AD: SciMLSensitivity continuous adjoint",
     "(InterpolatingAdjoint + EnzymeVJP)"])

# ---- coupling arrows (between tesseracts) ----
arrow((5.0, 2.15), (6.0, 2.15), "#b03a3a", r"$T_{pit}(t)$", ly=0.22)
arrow((6.0, 1.25), (5.0, 1.25), "#c96a1b", r"$q_{meat}(t)$", ly=-0.24)
ax.text(5.5, 1.72, "two-way\n(K=2, unrolled)", ha="center", va="center",
        fontsize=7.8, color="#555")

# ---- downward (primal) and upward (cotangent) crossings ----
arrow((2.2, 4.0), (2.2, 2.9), "#3b6ea5", r"$\varphi,\ u(t),\ q$", lx=-0.75)
arrow((3.4, 2.9), (3.4, 4.0), "#c96a1b", r"$T_{pit},\ m_{fuel}$", lx=0.85)
arrow((7.5, 4.0), (7.5, 2.9), "#3b6ea5", r"$\theta,\ T_{pit}(t)$", lx=-0.75)
arrow((8.7, 2.9), (8.7, 4.0), "#b03a3a", r"$T_{core},\ j_{evap},\ q,\ C$", lx=1.0)

# ---- gradient annotation ----
ax.text(5.5, 0.18,
        "one  jax.grad  of the composed log-density  =  4 VJP crossings of the boundary "
        "(2 continuous-adjoint solves + 2 forward-mode transposes)",
        ha="center", fontsize=9.2, color="#333", fontweight="bold")

fig.tight_layout()
out = REPO / "figures" / "composition_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
