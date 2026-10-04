# The top surface by multi-view stereo. A visual hull can't see a flat top come down: a point just above a slab still
# falls on the slab in every photo. The meat's texture can: per column of the footprint, a small horizontal patch is
# swept down from the hull's top, sampled in the photos that look down on it, and the height where they agree best
# (normalized cross-correlation of the patches' lightness) is the top; everything above it is carved. Where the
# photos don't agree well enough anywhere (no texture, a steep side) the hull's top stands.

# bilinear lightness at the continuous image point (x, y) (pixel (r, c) centred at (c − ½, r − ½)); NaN off the photo
@inline function sample(L::AbstractMatrix, x, y)
    h, w = size(L)
    cx = x + 0.5; cy = y + 0.5
    c0 = floor(Int, cx); r0 = floor(Int, cy)
    (1 <= c0 < w && 1 <= r0 < h) || return NaN
    fx = cx - c0; fy = cy - r0
    return (1 - fy) * ((1 - fx) * L[r0, c0] + fx * L[r0, c0 + 1]) + fy * ((1 - fx) * L[r0 + 1, c0] + fx * L[r0 + 1, c0 + 1])
end

@inline function project_xy(p::Proj, x, y, z)
    dx, dy, dz = x - p.C[1], y - p.C[2], z - p.C[3]
    X = p.R[1] * dx + p.R[2] * dy + p.R[3] * dz
    Y = p.R[4] * dx + p.R[5] * dy + p.R[6] * dz
    Z = p.R[7] * dx + p.R[8] * dy + p.R[9] * dz
    Z <= 0 && return (NaN, NaN)
    return (p.f * X / Z + p.cx, p.f * Y / Z + p.cy)
end

function ncc(a, b)
    ma = mean(a); mb = mean(b)
    num = 0.0; da = 0.0; db = 0.0
    @inbounds for i in eachindex(a)
        x = a[i] - ma; y = b[i] - mb
        num += x * y; da += x * x; db += y * y
    end
    return (da > 0 && db > 0) ? num / sqrt(da * db) : -1.0
end

"""
    refine_top!(occ, g, rig, masks; depth=25.0, half=4.0, n=7, minscore=0.6, minviews=3, el_min=0.5)

Lower each column's top to where the photos looking down on it (the sine of their elevation above `el_min`, at least
`minviews`) agree: a horizontal patch `2half` mm across on an `n × n` grid, swept from the hull's top down by up to
`depth` mm (1 mm steps, then 0.2 mm), scored by the mean NCC against the steepest view; kept when the score reaches
`minscore`. The heights are median-filtered over 5 × 5 columns before carving. Returns the fraction of columns
refined, the mean lowering [mm] and the top's height per column (the hull's where not refined; NaN off the meat).
"""
function refine_top!(occ::BitArray{3}, g::VoxelGrid, rig::Rig, masks::Vector{PhotoMask}; depth=25.0, half=4.0, n=7,
                     minscore=0.6, minviews=3, el_min=0.5)
    projs = [Proj(cam, size(m.meat)...) for (cam, m) in zip(rig.cams, masks)]
    Ls = [m.lab[:, :, 1] for m in masks]
    nx, ny, nz = g.dims
    ktop = zeros(Int, nx, ny)
    for I in findall(occ)
        i, j, k = Tuple(I); ktop[i, j] = max(ktop[i, j], k)
    end
    zhull = [ktop[i, j] == 0 ? NaN : g.origin[3] + ktop[i, j] * g.h for i in 1:nx, j in 1:ny]
    znew = fill(NaN, nx, ny)
    offs = range(-half, half; length=n)
    Threads.@threads for j in 1:ny
        ref = zeros(n * n); oth = zeros(n * n)
        for i in 1:nx
            ktop[i, j] == 0 && continue
            x, y, _ = cell_centre(g, i, j, 1); zt = zhull[i, j]
            # the views looking down on this column's top, the steepest first
            vs = Tuple{Float64,Int}[]
            for (v, cam) in enumerate(rig.cams)
                d = cam.C .- (x, y, zt); sinel = d[3] / norm(d)
                sinel > el_min || continue
                r, c = pixel(projs[v], x, y, zt)
                (r == 0 || !masks[v].meat[r, c]) && continue
                push!(vs, (sinel, v))
            end
            length(vs) >= minviews || continue
            sort!(vs; rev=true)
            function score(z)
                vr = vs[1][2]; k = 0
                for a in offs, b in offs
                    k += 1
                    u, w = project_xy(projs[vr], x + a, y + b, z); ref[k] = sample(Ls[vr], u, w)
                end
                any(isnan, ref) && return -1.0
                tot = 0.0; m = 0
                for (_, v) in vs[2:end]
                    k = 0
                    for a in offs, b in offs
                        k += 1
                        u, w = project_xy(projs[v], x + a, y + b, z); oth[k] = sample(Ls[v], u, w)
                    end
                    any(isnan, oth) && continue
                    tot += ncc(ref, oth); m += 1
                end
                return m >= minviews - 1 ? tot / m : -1.0
            end
            zlo = max(g.origin[3], zt - depth)
            best, zb = -Inf, zt
            for z in zt:-1.0:zlo
                s = score(z); s > best && ((best, zb) = (s, z))
            end
            for z in (zb + 1.0):-0.2:(zb - 1.0)
                (z > zt || z < zlo) && continue
                s = score(z); s > best && ((best, zb) = (s, z))
            end
            best >= minscore && (znew[i, j] = zb)
        end
    end
    # a 5 × 5 median of the refined heights, then carve above them; the top map keeps the hull's top elsewhere
    lowered = Float64[]
    top = copy(zhull)
    for j in 1:ny, i in 1:nx
        isnan(znew[i, j]) && continue
        nb = [znew[a, b] for a in max(1, i - 2):min(nx, i + 2), b in max(1, j - 2):min(ny, j + 2) if !isnan(znew[a, b])]
        zm = median(nb)
        kmax = floor(Int, (zm - g.origin[3]) / g.h + 0.5)                # cells whose centre lies below the top
        for k in (kmax + 1):nz
            occ[i, j, k] = false
        end
        push!(lowered, zhull[i, j] - zm); top[i, j] = zm
    end
    ncol = count(>(0), ktop)
    return length(lowered) / max(ncol, 1), isempty(lowered) ? 0.0 : mean(lowered), top
end
