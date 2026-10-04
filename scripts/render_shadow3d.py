"""3-D renderings of the trimmed brisket flat on the planned cook (figures/shadow3d/).

The 3-D shadow's cut (scripts/shadow3d.jl: a flat tapering in thickness and width from its thick end, a fat cap on
top, half of it solved across its width) drawn in 3-D:

- shadow3d_cutaway.png: the flat cut lengthwise through its centre (the solved half), temperature and moisture on
  the top, the cut face and the thick end at 2 h, at the wrap and at serve; the true cold spot marked at serve.
- shadow3d_brisket.png: the flat at serve as it comes off the smoker (the whole cut, the surface coloured by the
  model's lightness L*: Maillard bark and smoke deposition) and cut lengthwise (the bark, the fat cap, the smoke
  ring at its modelled depth under each face, the cooked interior).
- shadow3d_cook.gif: the lengthwise temperature cutaway through the cook, the camera turning.

The height is drawn twice its scale (a 2.5–5.5 cm flat against 33 cm of length); every figure says so. The
coordinates are the cut's raw (material) ones: the shrinkage is in the report and the plan figure. Every image is
supersampled (drawn at twice its resolution, downscaled), the faces' cells unbordered.

Usage (from app/):
    uv run python ../scripts/render_shadow3d.py
    uv run python ../scripts/render_shadow3d.py --plan schedule_robust.json --frames 56
"""

import argparse
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from PIL import Image
from shadow3d_common import HR, REPO, coords_grid, face_state, field_at, mirror_width, simulate

OUT = REPO / "figures" / "shadow3d"
EXAG = 2.0          # the height drawn twice its scale
UP = 3              # bilinear refinement of every face's node grid, for a smooth surface
VIEW = dict(elev=24, azim=128)   # from the thick end's side, facing the cut
SUPER = 2


def upsample(v, n=UP):
    """Bilinear refinement of a 2-D array of node values (n points per interval), the nodes kept."""
    def along(a, axis):
        m = a.shape[axis]
        t = np.linspace(0, m - 1, (m - 1) * n + 1)
        i = np.clip(np.floor(t).astype(int), 0, m - 2)
        w = t - i
        a0 = np.take(a, i, axis=axis); a1 = np.take(a, i + 1, axis=axis)
        sh = [1] * a.ndim; sh[axis] = -1
        return a0 * (1 - w.reshape(sh)) + a1 * w.reshape(sh)
    return along(along(np.asarray(v, dtype=float), 0), 1)


def render_supersampled(fig, dpi, tight=True):
    """The figure drawn at SUPER x `dpi` and downscaled to `dpi` with a Lanczos filter (smooth face edges)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SUPER * dpi, bbox_inches="tight" if tight else None)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    return im.resize((im.width // SUPER, im.height // SUPER), Image.LANCZOS)


def surface(ax, A, B, C, rgba, z=1):
    """One face, its cells unbordered and unaliased, painted at depth order `z` (the axes' own sort is off: faces
    meeting along an edge would otherwise paint over each other cell by cell)."""
    ax.computed_zorder = False
    return ax.plot_surface(A, B, C, facecolors=rgba, rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
                           zorder=z)


# --- the half cut, cut lengthwise ----------------------------------------------------------------------------------

def half_faces(sim, F):
    """The faces of the solved half seen from the cut side (the half at negative width): the top, the cut face on the
    centre plane and the thick end, each as (length, width, drawn height, node values)."""
    S, Y, Z = coords_grid(sim)
    return [(S[:, :, -1], -Y[:, :, -1], EXAG * Z[:, :, -1], F[:, :, -1]),
            (S[0, :, :], -Y[0, :, :], EXAG * Z[0, :, :], F[0, :, :]),
            (S[:, 0, :], np.zeros_like(S[:, 0, :]), EXAG * Z[:, 0, :], F[:, 0, :])]


def frame_half(ax, sim, title):
    g = sim["geometry"]
    ax.set_box_aspect((100 * g["length"], 50 * g["width_thick"], EXAG * 100 * g["thickness_thick"]))
    ax.set_xlim(0, 100 * g["length"]); ax.set_ylim(-50 * g["width_thick"], 0); ax.set_zlim(0, EXAG * 100 * g["thickness_thick"])
    ax.view_init(**VIEW)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=-6)


def draw_half(ax, sim, F, color_of):
    for z, (A, B, C, V) in enumerate(half_faces(sim, F)):
        surface(ax, upsample(A), upsample(B), upsample(C), color_of(upsample(V)), z=z + 1)


def snapshots(sim):
    t = sim["t"] / HR
    out = [(int(np.searchsorted(t, 2.0)), "2 h")]
    tw = sim["decisions"]["t_wrap"] / HR
    if tw < 1e4:
        k = int(np.searchsorted(t, tw)); out.append((k, f"wrap ({t[k]:.1f} h)"))
    out.append((len(t) - 1, f"serve ({t[-1]:.0f} h)"))
    return out


def plot_cutaway(sim):
    snaps = snapshots(sim)
    rows = (("T_field", "temperature [°C]", "inferno", lambda a: a - 273.15),
            ("W_field", "moisture [kg/kg db]", "viridis", lambda a: a))
    fig = plt.figure(figsize=(5.4 * len(snaps) + 1.2, 3.4 * len(rows) + 0.8))
    gs = fig.add_gridspec(len(rows), len(snaps) + 1, width_ratios=[1] * len(snaps) + [0.05], wspace=0.0, hspace=0.05)
    for r, (name, label, cmap, f) in enumerate(rows):
        fields = [f(field_at(sim, name, k)) for k, _ in snaps]
        norm = mcolors.Normalize(min(F.min() for F in fields), max(F.max() for F in fields))
        cm = plt.get_cmap(cmap)
        for c, ((k, title), F) in enumerate(zip(snaps, fields, strict=True)):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            draw_half(ax, sim, F, lambda v: cm(norm(v)))
            if name == "T_field" and c == len(snaps) - 1:
                # the cold spot: the coldest point on the faces the camera sees (its twin across the centre plane is
                # the grid's own), drawn above the surfaces
                best = None
                for A, B, C, V in half_faces(sim, F):
                    q = np.unravel_index(np.argmin(V), V.shape)
                    if best is None or V[q] < best[0]:
                        best = (V[q], A[q], B[q], C[q])
                ax.scatter([best[1]], [best[2]], [best[3]], s=90, facecolors="none", edgecolors="cyan", linewidths=2.2,
                           depthshade=False, zorder=10)
            frame_half(ax, sim, title if r == 0 else "")
        cax = fig.add_subplot(gs[r, -1])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, label=label)
    g = sim["geometry"]
    fig.suptitle(f"The brisket flat cut lengthwise through its centre ({100 * g['length']:.0f} cm long, "
                 f"{100 * g['thickness_thick']:.1f} → {100 * g['thickness_thin']:.1f} cm thick, the thick end nearest; "
                 f"height drawn ×{EXAG:.0f}); cyan: the cold spot at serve", fontsize=11)
    render_supersampled(fig, 130).save(OUT / "shadow3d_cutaway.png")
    plt.close(fig)
    print("wrote shadow3d_cutaway.png")


# --- the flat as it looks -------------------------------------------------------------------------------------------

BARK = mcolors.LinearSegmentedColormap.from_list("bark", ["#1e120c", "#4a2c1c", "#7a4a2e", "#b07a55", "#c99a7a"])
RING = np.array(mcolors.to_rgba("#c8566a"))
INTERIOR = np.array(mcolors.to_rgba("#8f6f62"))
CAP = np.array(mcolors.to_rgba("#eadfc8"))


def lstar_faces(sim, k):
    lm = sim["lstar_map"]
    L = lambda f: (lm["L0_bark"] - lm["dL_B"] * face_state(sim, "bark_dist", k, f)
                   - lm["dL_S"] * face_state(sim, "smoke_dist", k, f))
    lo, hi = lm["L0_bark"] - lm["dL_B"] - lm["dL_S"], lm["L0_bark"]
    rgba = lambda v: BARK(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    return {f: L(f) for f in ("top", "side", "thick", "thin")}, rgba


def plot_brisket(sim):
    S, Y, Z = coords_grid(sim)
    k = len(sim["t"]) - 1
    L, rgba = lstar_faces(sim, k)
    g = sim["geometry"]
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.0)

    # the whole flat off the smoker: the top over the full width, the near side, the thick end
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    Sf, Yf, Zf = (mirror_width(S[:, :, -1]), mirror_width(Y[:, :, -1], negate=True), mirror_width(Z[:, :, -1]))
    surface(ax, upsample(Sf), upsample(Yf), EXAG * upsample(Zf), rgba(upsample(mirror_width(L["top"]))), z=1)
    St, Yt, Zt = mirror_width(S[0, :, :], axis=0), mirror_width(Y[0, :, :], axis=0, negate=True), mirror_width(Z[0, :, :], axis=0)
    surface(ax, upsample(St), upsample(Yt), EXAG * upsample(Zt), rgba(upsample(mirror_width(L["thick"], axis=0))), z=2)
    surface(ax, upsample(S[:, -1, :]), upsample(Y[:, -1, :]), EXAG * upsample(Z[:, -1, :]), rgba(upsample(L["side"])), z=3)
    ax.set_box_aspect((100 * g["length"], 100 * g["width_thick"], EXAG * 100 * g["thickness_thick"]))
    ax.set_xlim(0, 100 * g["length"]); ax.set_ylim(-50 * g["width_thick"], 50 * g["width_thick"])
    ax.set_zlim(0, EXAG * 100 * g["thickness_thick"])
    ax.view_init(elev=26, azim=128); ax.set_axis_off()
    ax.set_title("off the smoker", fontsize=10, pad=-6)

    # cut lengthwise: the bark on the outer faces, the cut face drawn finely from the model's states
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    surface(ax, upsample(S[:, :, -1]), -upsample(Y[:, :, -1]), EXAG * upsample(Z[:, :, -1]), rgba(upsample(L["top"])), z=1)
    surface(ax, upsample(S[0, :, :]), -upsample(Y[0, :, :]), EXAG * upsample(Z[0, :, :]), rgba(upsample(L["thick"])), z=2)
    ring = {f: 100.0 * face_state(sim, "ring_dist", k, f) for f in ("top", "grate", "thick", "thin")}   # cm
    s_nodes = S[:, 0, 0]; H_nodes = Z[:, 0, -1]
    sf = np.linspace(s_nodes[0], s_nodes[-1], 330)
    Hf = np.interp(sf, s_nodes, H_nodes)
    cap = 100.0 * g["cap"]
    # the rows conform to the cap's underside (a row on it along the whole length), each quad coloured at its centre
    lean = np.linspace(0.0, 1.0, 106)[:, None] * (Hf - cap)[None, :]
    over = (Hf - cap)[None, :] + np.linspace(0.0, 1.0, 15)[1:, None] * cap
    ZZ = np.concatenate([lean, over], axis=0).T
    SS = np.repeat(sf[:, None], ZZ.shape[1], axis=1)
    mid = lambda a: np.pad(0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:]), ((0, 1), (0, 1)), mode="edge")
    Sc, Zc, Hc = mid(SS), mid(ZZ), mid(np.repeat(Hf[:, None], ZZ.shape[1], axis=1))
    sc = Sc[:, 0]
    ring_top = np.interp(sc, s_nodes, ring["top"][:, 0])[:, None]
    ring_grate = np.interp(sc, s_nodes, ring["grate"][:, 0])[:, None]
    ring_thick = float(np.mean(ring["thick"][0, :])); ring_thin = float(np.mean(ring["thin"][0, :]))
    d_top = Hc - Zc; d_grate = Zc; d_thick = Sc - s_nodes[0]; d_thin = s_nodes[-1] - Sc
    col = np.tile(INTERIOR, SS.shape + (1,))
    col[(d_grate <= ring_grate) | (d_thick <= ring_thick) | (d_thin <= ring_thin) | (d_top <= ring_top)] = RING
    col[d_top <= cap] = CAP                                          # the fat cap over the meat (the ring lies under it)
    col[(d_top <= 0.07) | (d_grate <= 0.07) | (d_thick <= 0.07) | (d_thin <= 0.07)] = BARK(0.12)
    surface(ax, SS, np.zeros_like(SS), EXAG * ZZ, col, z=3)
    frame_half(ax, sim, "cut lengthwise")
    fig.suptitle(f"The flat at serve: the surface by the model's L* (mean {sim['l_star'][-1]:.0f}; the raw rub "
                 f"{sim['lstar_map']['L0_bark']:.0f}), a {1000 * g['cap']:.0f} mm fat cap, the smoke ring "
                 f"{10 * float(np.mean(ring['grate'])):.1f} mm under the grate side\n"
                 f"(height drawn ×{EXAG:.0f}; the colours illustrate the modelled states, they are not a photograph)",
                 fontsize=10)
    render_supersampled(fig, 140).save(OUT / "shadow3d_brisket.png")
    plt.close(fig)
    print("wrote shadow3d_brisket.png")


def plot_animation(sim, frames, fps):
    t = sim["t"] / HR
    ks = np.unique(np.linspace(0, len(t) - 1, frames).round().astype(int))
    Fs = [field_at(sim, "T_field", k) - 273.15 for k in ks]
    norm = mcolors.Normalize(min(F.min() for F in Fs), max(F.max() for F in Fs))
    cm = plt.get_cmap("inferno")
    fig = plt.figure(figsize=(9.0, 5.0))
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.03, top=0.87)
    ax = fig.add_subplot(111, projection="3d")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, shrink=0.55, pad=0.02, label="temperature [°C]")
    dec = sim["decisions"]

    def phase(th):
        if th < sim.get("t_load", -1e9) / HR:
            return "preheating (meat off)"
        if th >= dec["t_pull"] / HR:
            return "resting"
        return "wrapped" if th >= dec["t_wrap"] / HR else "in the smoke"

    frames_rgb = []
    for n in range(len(ks)):
        ax.cla()
        F = Fs[n]
        draw_half(ax, sim, F, lambda v: cm(norm(v)))
        frame_half(ax, sim, "")
        ax.view_init(elev=VIEW["elev"], azim=VIEW["azim"] - 10.0 + 22.0 * n / max(len(ks) - 1, 1))
        fig.suptitle(f"t = {t[ks[n]]:4.1f} h · {phase(t[ks[n]])} · deepest point {float(sim['Tcore'][ks[n]]) - 273.15:5.1f} °C · "
                     f"cold spot {F.min():5.1f} °C (height ×{EXAG:.0f})", fontsize=9, y=0.98)
        # Keep one canvas size as the camera moves; cropping to the smallest
        # tight frame clips other frames' colorbar labels.
        frames_rgb.append(render_supersampled(fig, 100, tight=False))
    plt.close(fig)
    frames_rgb = [f.quantize(colors=255, method=Image.Quantize.MEDIANCUT) for f in frames_rgb]
    frames_rgb[0].save(OUT / "shadow3d_cook.gif", save_all=True, append_images=frames_rgb[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
    print("wrote shadow3d_cook.gif")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--nl", type=int, default=12)
    ap.add_argument("--nw", type=int, default=6)
    ap.add_argument("--nt", type=int, default=8)
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("simulating the planned cook in 3-D ...")
    sim = simulate(args.plan, args.nl, args.nw, args.nt)
    plot_cutaway(sim)
    plot_brisket(sim)
    plot_animation(sim, args.frames, args.fps)


if __name__ == "__main__":
    main()
