"""Composition/boundary diagram for the writeup (criterion 1).

Run from app/:  uv run python -u ../scripts/render_diagram.py
Writes figures/composition_diagram.png. Pure matplotlib (no Julia).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, INK = "#256abf", "#eb6834", "#2e8b57", "#0b0b0b"
FILL_JAX, FILL_TX, FILL_JL = "#eaf1fb", "#fdeee6", "#e9f4ec"

# one data unit = one inch, so point sizes map to the layout directly
fig = plt.figure(figsize=(9.2, 8.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 9.2)
ax.set_ylim(0, 8.0)
ax.axis("off")


def box(x, y, w, h, fc, ec, title, lines, title_size=10.5, size=8.4, lw=1.4):
    """Draw a rounded box with a bold title and body lines stacked beneath it."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.10",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=1,
        )
    )
    if title:
        ax.text(
            x + w / 2,
            y + h - 0.28,
            title,
            ha="center",
            va="top",
            fontsize=title_size,
            fontweight="bold",
            color=INK,
            zorder=2,
        )
    for i, s in enumerate(lines):
        ax.text(
            x + 0.18,
            y + h - 0.68 - 0.30 * i,
            s,
            ha="left",
            va="top",
            fontsize=size,
            color=INK,
            zorder=2,
        )


def arrow(p, q, color, lw=1.8, style="-|>"):
    """Draw an arrow from p to q."""
    ax.add_patch(
        FancyArrowPatch(
            p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, zorder=3
        )
    )


def label(x, y, s, color, size=8.2, ha="center"):
    """Draw a small caption with a white backdrop."""
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        zorder=4,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.95},
    )


def centered(x, y, s, size, color=INK, bold=False):
    """Draw centred text at (x, y)."""
    ax.text(
        x,
        y,
        s,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontweight="bold" if bold else "normal",
        zorder=2,
    )


centered(
    4.6,
    7.75,
    "BrisketDynamics — one Tesseract around the whole physics, forward-mode derivatives",
    12.5,
)

# --- JAX layer ---------------------------------------------------------------
ax.text(0.2, 7.4, "JAX  (app/)", fontsize=11, fontweight="bold", color=BLUE, va="top")
box(
    0.2,
    5.5,
    4.2,
    1.55,
    FILL_JAX,
    BLUE,
    "Calibration on the real cook",
    [
        "NumPyro model_real: masked AR(1) likelihood",
        "Gauss–Newton / Fisher from the Jacobian (130 s)",
        "NUTS, dense metric seeded from the Fisher",
    ],
)
box(
    4.7,
    5.5,
    4.2,
    1.55,
    FILL_JAX,
    BLUE,
    "Planning & control",
    [
        "fullplan: length × wrap × pull × chimney × spritz × vents",
        "MPC: hourly replans incl. crutch, pull, chimney, spritz",
        "robust: E[J] over posterior draws (Optax)",
    ],
)
box(0.55, 4.45, 8.0, 0.5, "white", BLUE, "", [])
centered(
    4.45,
    4.70,
    "coupled_cook(θ, φ, u, grids, wrap, pull, chimney, spritz, n_meat)  —  one apply_tesseract call",
    9.2,
    bold=True,
)
arrow((2.3, 5.4), (2.3, 5.06), BLUE)
arrow((6.8, 5.4), (6.8, 5.06), BLUE)

# --- Tesseract boundary ------------------------------------------------------
ax.add_patch(
    FancyBboxPatch(
        (0.2, 2.95), 8.7, 1.0, boxstyle="round,pad=0.08", fc=FILL_TX, ec=ORANGE, lw=1.6
    )
)
centered(4.55, 3.68, "Tesseract boundary  ·  brisket-monolith", 10.5, ORANGE, bold=True)
centered(
    4.55,
    3.36,
    "apply · abstract_eval · jacobian_vector_product · vector_jacobian_product · jacobian",
    8.0,
)
centered(
    4.55,
    3.10,
    "tesseract-jax: jax.grad and jax.jacfwd through the same interface",
    8.0,
)
arrow((4.55, 4.35), (4.55, 4.03), ORANGE, style="<|-|>")

# --- Julia layer ---------------------------------------------------------------
ax.text(
    0.2,
    2.78,
    "Julia  (src/monolith.jl)",
    fontsize=11,
    fontweight="bold",
    color=GREEN,
    va="top",
)
box(0.2, 0.25, 8.7, 2.1, FILL_JL, GREEN, "", [])
centered(
    4.55, 2.15, "monolith_solve — one stiff FBDF solve of 166 ODEs", 10.5, bold=True
)
centered(
    4.55,
    1.85,
    "derivatives: ForwardDiff through the solve — JVP = one Dual solve, "
    "VJP / Jacobian = one chunked pass",
    8.2,
    GREEN,
)
box(
    0.45,
    0.4,
    2.1,
    1.15,
    "white",
    GREEN,
    "pit",
    ["T_pit, m_fuel, ignited", "burn ∝ airflow(u) · ignited"],
    title_size=9.5,
    size=7.8,
)
box(
    3.45,
    0.4,
    2.6,
    1.15,
    "white",
    GREEN,
    "meat",
    ["T, W, C, F × 40 nodes + B, S, r", "stall, condensation, shrinkage, bark, ring"],
    title_size=9.5,
    size=7.8,
)
box(
    6.55,
    0.4,
    2.1,
    1.15,
    "white",
    GREEN,
    "pit humidity",
    ["p_v = base + meat vapour", "dilution by the draft"],
    title_size=9.5,
    size=7.8,
)
arrow((2.65, 1.25), (3.35, 1.25), GREEN, lw=1.4)
arrow((3.35, 0.7), (2.65, 0.7), GREEN, lw=1.4)
arrow((6.15, 1.0), (6.45, 1.0), GREEN, lw=1.4, style="<|-|>")
label(3.0, 1.45, "T_pit, ṁ", GREEN, size=7.4)
label(3.0, 0.5, "q_meat", GREEN, size=7.4)
arrow((4.55, 2.87), (4.55, 2.45), ORANGE, style="<|-|>")

out = REPO / "figures" / "composition_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out.relative_to(REPO)}")
