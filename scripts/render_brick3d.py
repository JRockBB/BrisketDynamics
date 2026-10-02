"""3-D renderings of the brisket flat on the planned cook (figures/brick3d/).

The 3-D shadow's cut (scripts/shadow3d.jl: a flat of the plan's thickness, a 2:1 footprint carrying the 1-D
exposed area, one eighth solved by symmetry) rebuilt whole by its three symmetry planes and drawn in 3-D:

- brick3d_cutaway.png: the cut with a quarter of its footprint removed down to the centre planes, temperature and
  moisture on every visible face at 2 h, at the wrap and at serve; the true cold spot marked at serve.
- brick3d_sliced.png: the cut at serve as it looks sliced: the surface coloured by the model's lightness L*
  (Maillard bark and smoke deposition), the smoke ring at its modelled depth under each face, the cooked interior.
- brick3d_cook.gif: the temperature cutaway through the cook, the camera turning.

The thickness is drawn three times its scale (a 3.9 cm flat against 35 cm of length); every figure says so. The
coordinates are the cut's raw (material) ones: the shrinkage is in the 3-D shadow's report and plan figure.

Usage (from app/):
    uv run python ../scripts/render_brick3d.py
    uv run python ../scripts/render_brick3d.py --plan schedule_robust.json --frames 60
"""

import argparse
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brick3d_common import HR, REPO, simulate
from matplotlib import colors as mcolors
from PIL import Image

OUT = REPO / "figures" / "brick3d"
EXAG = 3.0          # the thickness drawn at three times its scale
UP = 3              # bilinear upsampling of the node grid on every face, for a smooth surface
VIEW = dict(elev=24, azim=38)


# --- the whole cut from its eighth ---------------------------------------------------------------------------------

def full_axes(sim):
    """The whole cut's node coordinates [cm]: thickness (x, drawn exaggerated), width (y), length (z)."""
    nx, ny, nz = sim["dims"]
    c = np.asarray(sim["coords"])
    x = 100.0 * c[list(range(nx)), 0]
    y = 100.0 * c[[nx * j for j in range(ny)], 1]
    z = 100.0 * c[[nx * ny * k for k in range(nz)], 2]
    m = lambda a: np.concatenate([-a[::-1], a[1:]])
    return m(x), m(y), m(z)


def full_field(sim, field, k):
    """A node field at save index k over the whole cut, (x, y, z), mirrored across the three symmetry planes."""
    nx, ny, nz = sim["dims"]
    f = np.asarray(field)[:, k].reshape((nx, ny, nz), order="F")
    f = np.concatenate([f[::-1], f[1:]], axis=0)
    f = np.concatenate([f[:, ::-1], f[:, 1:]], axis=1)
    return np.concatenate([f[:, :, ::-1], f[:, :, 1:]], axis=2)


def face_state(sim, dist, k, face):
    """A surface state per exposed node on one face of the eighth (top: x-high, side: y-high, end: z-high), mirrored
    over that face's full extent. The exposed entries come face by face in the mesh's order."""
    nx, ny, nz = sim["dims"]
    d = np.asarray(dist)[:, k]
    top, side = ny * nz, nx * nz
    if face == "top":
        a = d[:top].reshape((ny, nz), order="F")                 # (y, z)
        a = np.concatenate([a[::-1], a[1:]], axis=0)
        return np.concatenate([a[:, ::-1], a[:, 1:]], axis=1)
    if face == "side":
        a = d[top:top + side].reshape((nx, nz), order="F")       # (x, z)
        a = np.concatenate([a[::-1], a[1:]], axis=0)
        return np.concatenate([a[:, ::-1], a[:, 1:]], axis=1)
    a = d[top + side:top + side + nx * ny].reshape((nx, ny), order="F")   # end: (x, y)
    a = np.concatenate([a[::-1], a[1:]], axis=0)
    return np.concatenate([a[:, ::-1], a[:, 1:]], axis=1)


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
    return along(along(v, 0), 1)


# --- the visible faces of the cutaway --------------------------------------------------------------------------------

def cutaway_faces(x, y, z, F):
    """The faces seen from the +length, +width quadrant with the quarter (y > 0, z > 0) removed, each as plot
    coordinates (length, width, drawn thickness) and the node values on it: the top, the two outer faces facing the
    camera, the two cut planes through the centre."""
    i0y, i0z = len(y) // 2, len(z) // 2
    X = EXAG
    faces = []
    # the top (x = +L), split around the removed quarter
    for ys, zs in ((slice(0, len(y)), slice(0, i0z + 1)), (slice(0, i0y + 1), slice(i0z, len(z)))):
        Zg, Yg = np.meshgrid(z[zs], y[ys])
        faces.append((Zg, Yg, np.full_like(Zg, X * x[-1]), F[-1, ys, zs]))
    # the outer face at width +W/2 (where the length is negative) and at length +Len/2 (where the width is negative)
    Zg, Xg = np.meshgrid(z[: i0z + 1], x)
    faces.append((Zg, np.full_like(Zg, y[-1]), X * Xg, F[:, -1, : i0z + 1]))
    Yg, Xg = np.meshgrid(y[: i0y + 1], x)
    faces.append((np.full_like(Yg, z[-1]), Yg, X * Xg, F[:, : i0y + 1, -1]))
    # the cut planes: width 0 for positive length, length 0 for positive width
    Zg, Xg = np.meshgrid(z[i0z:], x)
    faces.append((Zg, np.zeros_like(Zg), X * Xg, F[:, i0y, i0z:]))
    Yg, Xg = np.meshgrid(y[i0y:], x)
    faces.append((np.zeros_like(Yg), Yg, X * Xg, F[:, i0y:, i0z]))
    return faces


def surface(ax, A, B, C, rgba):
    """One face, its cells unbordered and unaliased (an anti-aliased or bordered mesh shows seams or moire between
    its cells); the figures are supersampled instead (`save_supersampled`)."""
    return ax.plot_surface(A, B, C, facecolors=rgba, rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)


SUPER = 2   # the supersampling factor: drawn at SUPER x the resolution, downscaled with a Lanczos filter


def render_supersampled(fig, dpi):
    """The figure drawn at SUPER x `dpi` and downscaled to `dpi` (smooth face edges without mesh artifacts)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SUPER * dpi, bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    return im.resize((im.width // SUPER, im.height // SUPER), Image.LANCZOS)


def draw_faces(ax, faces, color_of):
    for A, B, C, V in faces:
        surface(ax, upsample(A), upsample(B), upsample(C), color_of(upsample(V)))


def frame_axes(ax, x, y, z, title):
    ax.set_box_aspect((z[-1] - z[0], y[-1] - y[0], EXAG * (x[-1] - x[0])))
    ax.set_xlim(z[0], z[-1]); ax.set_ylim(y[0], y[-1]); ax.set_zlim(EXAG * x[0], EXAG * x[-1])
    ax.view_init(**VIEW)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=-4)


# --- the figures ---------------------------------------------------------------------------------------------------

def snapshots(sim):
    t = sim["t"] / HR
    out = [(int(np.searchsorted(t, 2.0)), "2 h")]
    tw = sim["decisions"]["t_wrap"] / HR
    if tw < 1e4:
        k = int(np.searchsorted(t, tw)); out.append((k, f"wrap ({t[k]:.1f} h)"))
    out.append((len(t) - 1, f"serve ({t[-1]:.0f} h)"))
    return out


def plot_cutaway(sim):
    x, y, z = full_axes(sim)
    snaps = snapshots(sim)
    rows = (("T_field", "temperature [°C]", "inferno", lambda a: a - 273.15),
            ("W_field", "moisture [kg/kg db]", "viridis", lambda a: a))
    fig = plt.figure(figsize=(5.2 * len(snaps) + 1.2, 3.6 * len(rows) + 0.8))
    gs = fig.add_gridspec(len(rows), len(snaps) + 1, width_ratios=[1] * len(snaps) + [0.05], wspace=0.02, hspace=0.05)
    for r, (name, label, cmap, f) in enumerate(rows):
        fields = [f(full_field(sim, sim[name], k)) for k, _ in snaps]
        norm = mcolors.Normalize(min(F.min() for F in fields), max(F.max() for F in fields))
        cm = plt.get_cmap(cmap)
        for c, ((k, title), F) in enumerate(zip(snaps, fields, strict=True)):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            draw_faces(ax, cutaway_faces(x, y, z, F), lambda v: cm(norm(v)))
            if name == "T_field" and c == len(snaps) - 1:
                # the cold spot: the coldest point among the faces the camera sees (the cut is symmetric, so the
                # grid's minimum has a twin on a visible corner or edge), drawn above the surfaces
                best = None
                for A, B, C, V in cutaway_faces(x, y, z, F):
                    q = np.unravel_index(np.argmin(V), V.shape)
                    if best is None or V[q] < best[0]:
                        best = (V[q], A[q], B[q], C[q])
                ax.scatter([best[1]], [best[2]], [best[3]], s=90, facecolors="none", edgecolors="cyan", linewidths=2.2,
                           depthshade=False, zorder=10)
                ax.computed_zorder = False
            frame_axes(ax, x, y, z, title if r == 0 else "")
        cax = fig.add_subplot(gs[r, -1])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax, label=label)
    fig.suptitle("The brisket flat in 3-D, a quarter cut away to the centre planes "
                 f"({100 * sim['geometry']['thickness']:.1f} × {100 * sim['geometry']['width']:.1f} × "
                 f"{100 * sim['geometry']['length']:.1f} cm; thickness drawn ×{EXAG:.0f}); cyan: the cold spot at serve",
                 fontsize=11)
    render_supersampled(fig, 130).save(OUT / "brick3d_cutaway.png")
    plt.close(fig)
    print("wrote brick3d_cutaway.png")


# the look of the cut: a raw rub's L* to a finished bark's, the smoke ring's pink, the cooked interior
BARK = mcolors.LinearSegmentedColormap.from_list("bark", ["#1e120c", "#4a2c1c", "#7a4a2e", "#b07a55", "#c99a7a"])
RING = np.array(mcolors.to_rgba("#c8566a"))
INTERIOR = np.array(mcolors.to_rgba("#8f6f62"))


def plot_sliced(sim):
    x, y, z = full_axes(sim)
    k = len(sim["t"]) - 1
    lm = sim["lstar_map"]                                        # the model's L* map: L0 − dL_B·bark − dL_S·smoke
    lstar = {f: lm["L0_bark"] - lm["dL_B"] * face_state(sim, sim["bark_dist"], k, f)
                - lm["dL_S"] * face_state(sim, sim["smoke_dist"], k, f) for f in ("top", "side", "end")}
    ring = {f: 100.0 * face_state(sim, sim["ring_dist"], k, f) for f in ("top", "side", "end")}   # cm
    lo, hi = lm["L0_bark"] - lm["dL_B"] - lm["dL_S"], lm["L0_bark"]
    bark_rgba = lambda v: BARK(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
    i0y, i0z = len(y) // 2, len(z) // 2
    X = EXAG
    fig = plt.figure(figsize=(11, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    faces = []                                                   # (length, width, drawn thickness, rgba), upsampled
    for ys, zs in ((slice(0, len(y)), slice(0, i0z + 1)), (slice(0, i0y + 1), slice(i0z, len(z)))):
        Zg, Yg = np.meshgrid(z[zs], y[ys])
        faces.append((upsample(Zg), upsample(Yg), np.full(upsample(Zg).shape, X * x[-1]), bark_rgba(upsample(lstar["top"][ys, zs]))))
    Zg, Xg = np.meshgrid(z[: i0z + 1], x)
    faces.append((upsample(Zg), np.full(upsample(Zg).shape, y[-1]), X * upsample(Xg), bark_rgba(upsample(lstar["side"][:, : i0z + 1]))))
    Yg, Xg = np.meshgrid(y[: i0y + 1], x)
    faces.append((np.full(upsample(Yg).shape, z[-1]), upsample(Yg), X * upsample(Xg), bark_rgba(upsample(lstar["end"][:, : i0y + 1]))))

    # a cut plane finely sampled: the smoke ring where a point lies within the ring depth of its nearest face (the
    # top or bottom by the top face's ring at that spot along the cut, the outer face it meets by that face's ring at
    # that height), the bark a thin line along the faces, the cooked interior elsewhere
    xf = np.linspace(x[0], x[-1], 90)

    def cut_plane(along, ring_top_line, ring_outer_by_x):
        sf = np.linspace(along[0], along[-1], 260)
        S, Xs = np.meshgrid(sf, xf)
        ring_top = np.interp(sf, along, ring_top_line)[None, :]
        ring_outer = np.interp(xf, x, ring_outer_by_x)[:, None]
        depth_tb = x[-1] - np.abs(Xs)
        depth_outer = along[-1] - S
        rgba = np.tile(INTERIOR, S.shape + (1,))
        rgba[(depth_tb <= ring_top) | (depth_outer <= ring_outer)] = RING
        rgba[(depth_tb <= 0.06) | (depth_outer <= 0.06)] = BARK(0.12)
        return S, Xs, rgba

    # the plane width = 0 along the length (z > 0) meets the end face; the plane length = 0 along the width (y > 0)
    # meets the side face
    S, Xs, rgba = cut_plane(z[i0z:], ring["top"][i0y, i0z:], ring["end"][:, i0y])
    faces.append((S, np.zeros_like(S), X * Xs, rgba))
    S, Xs, rgba = cut_plane(y[i0y:], ring["top"][i0y:, i0z], ring["side"][:, i0z])
    faces.append((np.zeros_like(S), S, X * Xs, rgba))
    for A, B, C, rgba in faces:
        surface(ax, A, B, C, rgba)
    frame_axes(ax, x, y, z, "")
    ax.set_title(f"The flat at serve, sliced\n"
                 f"the surface by the model's L* (mean {sim['l_star'][-1]:.0f}; the raw rub {lm['L0_bark']:.0f}), "
                 f"the smoke ring {10 * float(np.mean(ring['top'])):.1f} mm under the top\n"
                 f"thickness drawn ×{EXAG:.0f}; the colours illustrate the modelled states, they are not a photograph",
                 fontsize=10)
    render_supersampled(fig, 140).save(OUT / "brick3d_sliced.png")
    plt.close(fig)
    print("wrote brick3d_sliced.png")


def plot_animation(sim, frames, fps):
    x, y, z = full_axes(sim)
    t = sim["t"] / HR
    ks = np.unique(np.linspace(0, len(t) - 1, frames).round().astype(int))
    Fs = [full_field(sim, sim["T_field"], k) - 273.15 for k in ks]
    norm = mcolors.Normalize(min(F.min() for F in Fs), max(F.max() for F in Fs))
    cm = plt.get_cmap("inferno")
    fig = plt.figure(figsize=(8.6, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, shrink=0.6, pad=0.02, label="temperature [°C]")
    dec = sim["decisions"]

    def phase(th):
        if th >= dec["t_pull"] / HR:
            return "resting"
        return "wrapped" if th >= dec["t_wrap"] / HR else "in the smoke"

    def draw(n):
        ax.cla()
        F = Fs[n]
        draw_faces(ax, cutaway_faces(x, y, z, F), lambda v: cm(norm(v)))
        frame_axes(ax, x, y, z, "")
        # a slow turn that keeps every drawn face well off edge-on
        ax.view_init(elev=VIEW["elev"], azim=VIEW["azim"] - 10.0 + 22.0 * n / max(len(ks) - 1, 1))
        i, j, kk = np.unravel_index(np.argmin(F), F.shape)
        ax.set_title(f"t = {t[ks[n]]:4.1f} h · {phase(t[ks[n]])} · centre {F[len(x) // 2, len(y) // 2, len(z) // 2]:5.1f} °C · "
                     f"cold spot {F[i, j, kk]:5.1f} °C (thickness ×{EXAG:.0f})", fontsize=9)
        return []

    frames_rgb = []
    for n in range(len(ks)):
        draw(n)
        frames_rgb.append(render_supersampled(fig, 100))
    plt.close(fig)
    w = min(f.width for f in frames_rgb); h = min(f.height for f in frames_rgb)
    frames_rgb = [f.crop((0, 0, w, h)).quantize(colors=255, method=Image.Quantize.MEDIANCUT) for f in frames_rgb]
    frames_rgb[0].save(OUT / "brick3d_cook.gif", save_all=True, append_images=frames_rgb[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
    print("wrote brick3d_cook.gif")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default="schedule_robust.json")
    ap.add_argument("--nx", type=int, default=6)
    ap.add_argument("--ny", type=int, default=8)
    ap.add_argument("--nz", type=int, default=12)
    ap.add_argument("--frames", type=int, default=56)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("simulating the planned cook in 3-D ...")
    sim = simulate(args.plan, args.nx, args.ny, args.nz)
    plot_cutaway(sim)
    plot_sliced(sim)
    plot_animation(sim, args.frames, args.fps)


if __name__ == "__main__":
    main()
