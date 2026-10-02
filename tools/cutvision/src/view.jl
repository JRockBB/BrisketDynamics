# A look at a reconstruction: rays from a pinhole camera stepped through the label grid (Amanatides–Woo) to the first
# solid cell, coloured by what it is (lean, fat, bone) and shaded by the face it entered through. `keep` hides
# cells (a cut-away: e.g. everything past a plane) so the inside shows.

const VIEW_COLOURS = Dict(LEAN => (0.62, 0.17, 0.19), FAT => (0.93, 0.89, 0.80), BONE => (0.97, 0.97, 0.94))

"""
    render_voxels(g, L; C, target, f=1400.0, width=1200, height=900, up=(0, 0, 1), keep=(x, y, z) -> true)

The labels `L` on grid `g` seen from `C` looking at `target` (board frame, mm), on a light grey ground; `keep(x, y, z)`
false hides a cell (by its centre).
"""
function render_voxels(g::VoxelGrid, L::Array{Int8,3}; C, target, f=1400.0, width=1200, height=900, up=(0.0, 0.0, 1.0),
                       keep=(x, y, z) -> true)
    zc = normalize(collect(target) .- collect(C)); xc = normalize(cross(zc, collect(up))); yc = cross(zc, xc)
    Rt = hcat(xc, yc, zc)
    light = normalize([0.3, -0.4, 1.0])
    img = Matrix{RGB{N0f8}}(undef, height, width)
    lo = collect(g.origin); hi = lo .+ g.h .* collect(g.dims)
    solid(i, j, k) = (l = L[i, j, k]; l != EMPTY && keep(cell_centre(g, i, j, k)...))
    Threads.@threads for r in 1:height
        for c in 1:width
            d = Rt * [(c - 0.5 - width / 2) / f, (r - 0.5 - height / 2) / f, 1.0]; d = d / norm(d)
            col = (0.85, 0.85, 0.85); o = collect(C)
            # enter the grid's box
            t0, t1 = 0.0, Inf
            for a in 1:3
                if abs(d[a]) < 1e-12
                    (o[a] < lo[a] || o[a] > hi[a]) && (t0 = Inf)
                else
                    ta = (lo[a] - o[a]) / d[a]; tb = (hi[a] - o[a]) / d[a]
                    t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb))
                end
            end
            if t0 < t1
                p = o .+ (t0 + 1e-6) .* d
                idx = [clamp(floor(Int, (p[a] - lo[a]) / g.h) + 1, 1, g.dims[a]) for a in 1:3]
                step = [d[a] > 0 ? 1 : -1 for a in 1:3]
                tmax = [abs(d[a]) < 1e-12 ? Inf : ((lo[a] + (idx[a] - (d[a] < 0 ? 1 : 0)) * g.h) - o[a]) / d[a] for a in 1:3]
                tdel = [abs(d[a]) < 1e-12 ? Inf : g.h / abs(d[a]) for a in 1:3]
                face = argmin([abs(t0 - (d[a] > 0 ? (lo[a] - o[a]) / d[a] : (hi[a] - o[a]) / d[a])) for a in 1:3])
                while true
                    if solid(idx...)
                        nrm = zeros(3); nrm[face] = -step[face]
                        base = VIEW_COLOURS[L[idx...]]
                        sh = 0.55 + 0.45 * max(dot(nrm, light), 0.0)
                        col = base .* sh
                        break
                    end
                    face = argmin(tmax)
                    idx[face] += step[face]
                    (1 <= idx[face] <= g.dims[face]) || break
                    tmax[face] += tdel[face]
                end
            end
            img[r, c] = RGB{N0f8}(clamp.(col, 0, 1)...)
        end
    end
    return img
end
