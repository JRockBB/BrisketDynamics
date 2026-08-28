"""Mathematical composition diagram: the coupled differential equations.

Run from app/:  uv run python -u ../scripts/render_math_diagram.py
Writes figures/math_diagram.png — the companion to composition_diagram.png
showing WHAT is being solved rather than how it is packaged: each
component's equations in a box, with the exchanged signals as arrows.
Full derivations and constants: docs/model.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]

INK = "#0b0b0b"
MUT = "#52514e"
PIT_FC, PIT_EC = "#fdeadd", "#c96a1f"
MEAT_FC, MEAT_EC = "#e4eefb", "#256abf"
SUB_FC = "#ffffff"
OBJ_FC, OBJ_EC = "#e7f2e4", "#3d7a34"
IN_FC, IN_EC = "#f2f1ec", "#8a887f"
ARR_F = "#c96a1f"  # pit -> meat forcings
ARR_B = "#256abf"  # meat -> pit load
ARR_O = "#3d7a34"  # states -> objective

fig, ax = plt.subplots(figsize=(14.5, 9.9))
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 9.9)
ax.axis("off")

LH = 0.46  # equation line height


def box(x, y, w, h, fc, ec, title, lines, size=10.5, title_size=11.5, lw=1.6):
    """Rounded box with a bold title and mathtext body lines."""
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", fc=fc, ec=ec, lw=lw)
    )
    ax.text(
        x + w / 2,
        y + h - 0.32,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
    )
    for i, ln in enumerate(lines):
        ax.text(
            x + w / 2,
            y + h - 0.75 - LH * i,
            ln,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
        )


def arrow(p, q, color, label=None, lpos=None, lw=2.2, rad=0.0, size=9.5):
    """Labeled arrow; label gets a white pad so it never fights the line."""
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=16,
            color=color,
            lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        lx, ly = lpos if lpos else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.95,
            },
        )


# ---- inputs (top) ---------------------------------------------------------
box(
    0.35,
    7.9,
    2.5,
    1.0,
    IN_FC,
    IN_EC,
    "control input",
    [r"$u(t)\in[0,1]$  damper"],
    size=10,
)
box(
    3.25,
    7.9,
    3.1,
    1.0,
    IN_FC,
    IN_EC,
    "wrap decision",
    [r"$s(t)=\sigma\!\left((t - t_{wrap})/\delta_w\right)$"],
    size=10,
)

# ---- pit (left) -----------------------------------------------------------
box(
    0.35,
    4.4,
    5.0,
    2.9,
    PIT_FC,
    PIT_EC,
    "PIT — lumped ODEs (3 states)",
    [
        r"$C_{th}\,\dot T_p \,=\, \Delta H_{e\!f\!f}\,\dot m_b \;-\; UA\,(T_p - T_{amb}) \;-\; q_{meat}$",
        r"$\dot m = -\,\dot m_b\,,\qquad \dot m_b = r_{max}\; g(u)\; a\,\tanh(m/m_t)$",
        r"$\dot a \,=\, k_{spr}\, a\,(1-a)\;g(u)$",
        r"$g(u) = u_{leak} + (1-u_{leak})\,u\quad$ (leaky damper airflow)",
    ],
)

# ---- meat (right) ---------------------------------------------------------
box(
    6.6,
    2.45,
    7.55,
    6.65,
    MEAT_FC,
    MEAT_EC,
    "MEAT — 1-D PDE + kinetics ($x\\in[0,L]$, 162 ODEs after MOL)",
    [],
)
# interior transport
box(
    6.85,
    6.65,
    7.05,
    1.80,
    SUB_FC,
    MEAT_EC,
    "interior fields",
    [
        r"$\partial_t T = \alpha\,\partial_x^2 T\qquad\quad \partial_t C = k_C(T)\,(1-C)$",
        r"$\partial_t W = D_{e\!f\!f}\,\partial_x^2 W \,-\, k_e\,\mathrm{sp}_{\delta}\!\left(W - W_{eq}(T)\right)$",
        r"$\partial_t F = -\,k_f(T)\,F\qquad$ ($W_{eq}(T)$: myosin + actin steps)",
    ],
    size=10,
    lw=1.2,
)
# surface BC
box(
    6.85,
    4.35,
    7.05,
    2.0,
    SUB_FC,
    MEAT_EC,
    "surface boundary ($x = L$; wrap blend $s$)",
    [
        r"$j = (1-\eta_e\, s)\;k_m\,\varphi(W_s)\,\Delta\rho_v(T_s)\,,\qquad -\rho_d D_{e\!f\!f}\,\partial_x W|_s = j$",
        r"$k\,\partial_x T|_s = h_c\,(1-\eta_h\, s)\,(T_p - T_s) + \varepsilon(s)\,\sigma_{SB}\,(T_p^4 - T_s^4) - \lambda\,j$",
        r"$\varepsilon(s) = \varepsilon\,(1-s) + \varepsilon_{wrap}\,s\qquad q_{surf} = k\,\partial_x T|_s + \lambda\,j$",
    ],
    size=10,
    lw=1.2,
)
# bark states
box(
    6.85,
    2.60,
    7.05,
    1.45,
    SUB_FC,
    MEAT_EC,
    "bark states (surface)",
    [
        r"$\dot B = k_B(T_s)\,g(a_w)\,(1-B)\qquad\quad L^* = L_0^* - \Delta_B B - \Delta_S S$",
        r"$\dot S = k_{dep}\,\mathrm{wet}(a_w)\;\dot m_b\;(1-s)\,(1-S)$",
    ],
    size=10,
    lw=1.2,
)

# ---- objective (bottom) ---------------------------------------------------
box(
    0.35,
    0.3,
    13.8,
    1.7,
    OBJ_FC,
    OBJ_EC,
    "QUALITY OBJECTIVE  $J$  (KCBS weights; every term a smooth band)",
    [
        r"$J = 0.14\,\mathcal{A}(L^*) \;+\; 0.57\,\left[\mathcal{G}_{smoke}(S) + w_d\,(1-\mathrm{juiciness}(W,F))^2\right] \;+\; 0.29\,\mathcal{T}(C_0) \;+\; \mathrm{ops}(T_s, m, u)$",
        r"$+\;\; \mathcal{S}_{safety}:\ \ F_{leth} = \int 10^{(T_s - T_{ref})/z}\,dt \geq F_{req}\,,\quad$ danger-zone dwell $\leq 4\,$h",
    ],
    size=10.5,
)

# ---- coupling arrows ------------------------------------------------------
# u -> pit: adjacent boxes, straight vertical drop
arrow((1.6, 7.85), (1.6, 7.35), MUT, lw=2.4)
# wrap blend -> surface BC: through the open pit-meat corridor, entering
# at the sub-box title level (well above the T_p lane)
arrow((5.9, 7.85), (6.8, 6.25), MUT, label=r"$s(t)$", lpos=(5.95, 7.0))
# pit temperature -> surface BC (middle lane, clearly below s(t))
arrow((5.4, 6.0), (6.8, 5.5), ARR_F, label=r"$T_p(t)$", lpos=(6.05, 5.97))
# meat heat load -> pit (lower lane, parallel, no crossing)
arrow(
    (6.8, 4.95),
    (5.4, 5.35),
    ARR_B,
    label=r"$q_{meat}(t) = A\,q_{surf}$",
    lpos=(6.02, 4.88),
)
# burn rate -> smoke deposition
arrow((5.0, 4.35), (6.8, 3.45), ARR_F, label=r"$\dot m_b(t)$", lpos=(5.62, 3.72))
# states -> objective
arrow((2.85, 4.35), (2.85, 2.05), ARR_O, label=r"fuel $m$", lpos=(3.38, 3.15))
arrow((9.0, 2.32), (9.0, 2.08), ARR_O)
ax.text(
    9.45,
    2.22,
    r"$C_0,\,W,\,F,\,B,\,S,\,T_s$",
    ha="left",
    va="center",
    fontsize=9.5,
    color=ARR_O,
    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.95},
)

# Picard coupling note
ax.text(
    7.25,
    9.55,
    "coupling: whole-horizon Picard iteration, K = 2 unrolled passes — "
    "the model IS the unrolled map, so composed VJPs give exact gradients",
    ha="center",
    va="center",
    fontsize=9.5,
    color=MUT,
    style="italic",
)

fig.suptitle(
    "What the pipeline differentiates: the coupled equations "
    "(constants & derivations: docs/model.md)",
    fontsize=13,
    y=0.985,
)
fig.tight_layout(rect=(0, 0, 1, 0.965))
out = REPO / "figures" / "math_diagram.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")
