# What each voxel is made of: 1 lean, 2 fat, 3 bone (0 empty). The surface is labelled from the photos (each surface
# voxel's colour voted over the photos that see it, unoccluded and facing the camera); the cap's depth is measured on
# the steep faces bordering a fat top (the fat band's height, the surface's own record of the cap), a meat
# profile's default when there are none; the interior is fat within that depth of a fat surface that is its nearest
# surface; bone is placed by the meat's template in the shape's principal frame.

const EMPTY, LEAN, FAT, BONE = Int8(0), Int8(1), Int8(2), Int8(3)

"""
    MeatProfile(name, cap_mm, bone)

A meat's rules for what the photos can't see: the default cap depth [mm] (used when no steep fat face is measured)
and the bone template (`:none`, `:rib_rods`, `:blade_plate`).
"""
struct MeatProfile
    name::Symbol
    cap_mm::Float64
    bone::Symbol
end

const PROFILES = Dict(
    :brisket => MeatProfile(:brisket, 6.0, :none),
    :chuck => MeatProfile(:chuck, 3.0, :none),
    :pork_butt => MeatProfile(:pork_butt, 8.0, :blade_plate),
    :ribs => MeatProfile(:ribs, 3.0, :rib_rods),
)

# ---- the surface and its normals ----------------------------------------------------------------------------------

const NB6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

occupied(occ, i, j, k) = checkbounds(Bool, occ, i, j, k) && occ[i, j, k]

# occupied voxels with an empty face neighbour other than the board below
function surface_voxels(occ)
    S = CartesianIndex{3}[]
    for I in findall(occ)
        i, j, k = Tuple(I)
        any(d -> !(d == (0, 0, -1) && k == 1) && !occupied(occ, i + d[1], j + d[2], k + d[3]), NB6) && push!(S, I)
    end
    return S
end

# a separable Gaussian blur of the occupancy (σ in voxels)
function blur(occ, σ)
    r = ceil(Int, 3σ); w = [exp(-x^2 / (2σ^2)) for x in -r:r]; w ./= sum(w)
    A = Float64.(occ)
    for d in 1:3
        B = zeros(size(A))
        n = size(A, d)
        for I in CartesianIndices(A)
            s = 0.0
            for (t, wt) in zip(-r:r, w)
                J = Base.setindex(Tuple(I), clamp(I[d] + t, 1, n), d)
                # the board below the meat counts as meat for the normals (the floor is not a face)
                v = (d == 3 && I[d] + t < 1) ? A[Base.setindex(Tuple(I), 1, 3)...] : A[J...]
                s += wt * v
            end
            B[I] = s
        end
        A = B
    end
    return A
end

# trilinear value of the blurred occupancy at a board point (cells centred at their centres; 0 outside the grid)
function field(B, g::VoxelGrid, x, y, z)
    u = (x - g.origin[1]) / g.h + 0.5; v = (y - g.origin[2]) / g.h + 0.5; w = (z - g.origin[3]) / g.h + 0.5
    i = floor(Int, u); j = floor(Int, v); k = floor(Int, w)
    fu = u - i; fv = v - j; fw = w - k
    val(a, b, c) = checkbounds(Bool, B, a, b, c) ? B[a, b, c] : (c < 1 && checkbounds(Bool, B, a, b, 1) ? B[a, b, 1] : 0.0)
    s = 0.0
    for (di, wi) in ((0, 1 - fu), (1, fu)), (dj, wj) in ((0, 1 - fv), (1, fv)), (dk, wk) in ((0, 1 - fw), (1, fw))
        s += wi * wj * wk * val(i + di, j + dj, k + dk)
    end
    return s
end

# the outward normal at each surface voxel: the blurred occupancy's descending gradient (and the blurred field)
function normals(occ, S; σ=0.8)
    B = blur(occ, σ)
    return _normals(B, S), B
end

function _normals(B, S)
    n = size(B)
    g(I, d) = (Ip = Base.setindex(Tuple(I), min(I[d] + 1, n[d]), d); Im = Base.setindex(Tuple(I), max(I[d] - 1, 1), d);
               B[Ip...] - B[Im...])
    return [(v = -[g(I, 1), g(I, 2), g(I, 3)]; norm(v) > 0 ? v / norm(v) : [0.0, 0.0, 1.0]) for I in S]
end

# ---- visibility and the colour vote -------------------------------------------------------------------------------

# whether the segment from point X toward the camera centre C leaves the grid without crossing an occupied voxel
function unoccluded(g::VoxelGrid, occ, X, C)
    d = C .- X; L = norm(d); d = d ./ L
    step = 0.5g.h
    t = 1.5g.h
    while t < L
        p = X .+ t .* d
        i = floor(Int, (p[1] - g.origin[1]) / g.h) + 1; j = floor(Int, (p[2] - g.origin[2]) / g.h) + 1
        k = floor(Int, (p[3] - g.origin[3]) / g.h) + 1
        (1 <= i <= g.dims[1] && 1 <= j <= g.dims[2] && 1 <= k <= g.dims[3]) || return true
        occ[i, j, k] && return false
        t += step
    end
    return true
end

"""
    surface_labels(g, occ, rig, masks, isfat)

Each surface voxel's fat vote, sampled on the surface half a voxel out along its normal, over the photos that see it
(facing the camera by more than 15°, a steep face within 53°, unoccluded, landing on the meat), weighted by how
squarely, then averaged with the seen surface neighbours' (a single vote is noisy); the voxels, their normals, the
fat fraction (NaN unseen) and the blurred occupancy.
"""
function surface_labels(g::VoxelGrid, occ, rig::Rig, masks::Vector{PhotoMask}, isfat)
    S = surface_voxels(occ)
    N, B = normals(occ, S)
    projs = [Proj(cam, size(m.meat)...) for (cam, m) in zip(rig.cams, masks)]
    frac = fill(NaN, length(S))
    Threads.@threads for s in eachindex(S)
        X0 = collect(cell_centre(g, Tuple(S[s])...))
        X = X0 .+ 0.5g.h .* N[s]                                     # on the surface, not the voxel's centre inside it
        wf = 0.0; wt = 0.0
        for (v, cam) in enumerate(rig.cams)
            toC = cam.C .- X; cosang = dot(N[s], toC) / norm(toC)
            cosang > (abs(N[s][3]) < 0.5 ? 0.6 : 0.26) || continue  # steep faces: only views square to them
            unoccluded(g, occ, X0, cam.C) || continue
            r, c = pixel(projs[v], X...)
            r == 0 && continue
            m = masks[v]
            m.meat[r, c] || continue
            wt += cosang
            isfat(m.lab[r, c, 1], m.lab[r, c, 2], m.lab[r, c, 3]) && (wf += cosang)
        end
        wt > 0 && (frac[s] = wf / wt)
    end
    # one voxel's vote is noisy (texture, a pixel on a boundary): each takes the mean of its seen surface neighbours'
    at = Dict(Tuple(S[s]) => s for s in eachindex(S))
    smooth = similar(frac)
    for s in eachindex(S)
        i, j, k = Tuple(S[s]); acc = 0.0; n = 0
        for di in -1:1, dj in -1:1, dk in -1:1
            q = get(at, (i + di, j + dj, k + dk), 0)
            (q == 0 || isnan(frac[q])) && continue
            acc += frac[q]; n += 1
        end
        smooth[s] = n == 0 ? NaN : acc / n
    end
    return S, N, smooth, B
end

# ---- distances ----------------------------------------------------------------------------------------------------

# Felzenszwalb–Huttenlocher squared distance transform of one line (f the squared distances so far; BIG where none)
const BIG = 1e20
function _edt1!(f::AbstractVector{Float64}, out, v, z)
    n = length(f); k = 1; v[1] = 1; z[1] = -Inf; z[2] = Inf
    for q in 2:n
        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2q - 2v[k])
        while s <= z[k]
            k -= 1
            s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2q - 2v[k])
        end
        k += 1; v[k] = q; z[k] = s; z[k + 1] = Inf
    end
    k = 1
    for q in 1:n
        while z[k + 1] < q
            k += 1
        end
        out[q] = (q - v[k])^2 + f[v[k]]
    end
    return out
end

# the 2-D distance [cells] to the nearest `src` cell
function edt2(src::AbstractMatrix{Bool})
    D = map(b -> b ? 0.0 : BIG, src)
    for d in 1:2
        n = size(D, d)
        f = zeros(n); out = zeros(n); v = zeros(Int, n); z = zeros(n + 1)
        for o in 1:size(D, 3 - d)
            line = d == 1 ? view(D, :, o) : view(D, o, :)
            f .= line; _edt1!(f, out, v, z); line .= out
        end
    end
    return map(x -> x >= BIG / 2 ? Inf : sqrt(x), D)
end

"""
    edt(src)

The Euclidean distance [voxels] from every voxel to the nearest `src` voxel (Inf when there is none).
"""
function edt(src::AbstractArray{Bool,3})
    D = map(b -> b ? 0.0 : BIG, src)
    for d in 1:3
        n = size(D, d)
        f = zeros(n); out = zeros(n); v = zeros(Int, n); z = zeros(n + 1)
        for I in CartesianIndices(Base.setindex(size(D), 1, d))
            line = [Base.setindex(Tuple(I), q, d) for q in 1:n]
            for q in 1:n
                f[q] = D[line[q]...]
            end
            _edt1!(f, out, v, z)
            for q in 1:n
                D[line[q]...] = out[q]
            end
        end
    end
    return map(x -> x >= BIG / 2 ? Inf : sqrt(x), D)
end

# ---- the cap's depth ----------------------------------------------------------------------------------------------

"""
    cap_depth(g, occ, B, S, N, rig, masks, isfat; zstep=0.25, span=40.0, steep=0.35)

The fat band's depth on the steep faces, read from the photos below the voxels' resolution. Per column holding a
steep surface voxel, the surface is walked down from the column's top every `zstep` mm (at each height the point
along the face's horizontal normal where the blurred occupancy `B` crosses ½, its normal the field's gradient); the
face turns steep where |n_z| drops below `steep` (the edge of the up-facing top, the cap rule's own threshold); each
point's fat vote is averaged over the photos square to it (within 53°, unoccluded). The band runs from the edge down
to where the votes turn lean for a millimetre; columns whose edge reads lean, or with no lean within `span` mm, are
skipped. The trimmed mean (middle half) over the columns, and their count.
"""
function cap_depth(g::VoxelGrid, occ, B, S, N, rig::Rig, masks::Vector{PhotoMask}, isfat; zstep=0.25, span=40.0, steep=0.35)
    projs = [Proj(cam, size(m.meat)...) for (cam, m) in zip(rig.cams, masks)]
    best = Dict{Tuple{Int,Int},Int}()
    for s in eachindex(S)
        abs(N[s][3]) < steep || continue
        key = (S[s][1], S[s][2])
        (!haskey(best, key) || S[s][3] > S[best[key]][3]) && (best[key] = s)
    end
    cols = collect(values(best))
    ktop = Dict{Tuple{Int,Int},Int}()
    for I in findall(occ)
        i, j, k = Tuple(I); ktop[(i, j)] = max(get(ktop, (i, j), 0), k)
    end
    bands = fill(NaN, length(cols))
    ε = 0.3
    grad(x, y, z) = -[field(B, g, x + ε, y, z) - field(B, g, x - ε, y, z), field(B, g, x, y + ε, z) - field(B, g, x, y - ε, z),
                      field(B, g, x, y, z + ε) - field(B, g, x, y, z - ε)]
    Threads.@threads for q in eachindex(cols)
        s = cols[q]; i, j, _ = Tuple(S[s])
        c = cell_centre(g, i, j, 1)
        n = N[s]; nxy = (n[1], n[2]) ./ max(hypot(n[1], n[2]), 1e-9)
        X0 = collect(cell_centre(g, Tuple(S[s])...))
        zt = g.origin[3] + ktop[(i, j)] * g.h + g.h
        ze = NaN; zlean = NaN; run = 0; need = ceil(Int, 1.0 / zstep)
        for z in zt:-zstep:max(g.origin[3] + zstep, zt - span)
            # the surface at this height along the column's normal: where the blurred occupancy crosses ½
            lo, hi = -4g.h, 4g.h
            (field(B, g, c[1] + lo * nxy[1], c[2] + lo * nxy[2], z) >= 0.5 &&
             field(B, g, c[1] + hi * nxy[1], c[2] + hi * nxy[2], z) < 0.5) || continue
            for _ in 1:20
                mid = (lo + hi) / 2
                field(B, g, c[1] + mid * nxy[1], c[2] + mid * nxy[2], z) >= 0.5 ? (lo = mid) : (hi = mid)
            end
            x = c[1] + lo * nxy[1]; y = c[2] + lo * nxy[2]
            gn = grad(x, y, z); norm(gn) > 0 || continue
            gn = gn / norm(gn)
            if isnan(ze)
                abs(gn[3]) < steep || continue                      # still on the up-facing top
                ze = z
            end
            k = 0; f = 0
            for (v, cam) in enumerate(rig.cams)
                d = cam.C .- (x, y, z)
                dot(gn, d) / norm(d) > 0.6 || continue
                unoccluded(g, occ, X0, cam.C) || continue
                r, cc = pixel(projs[v], x, y, z)
                (r == 0 || !masks[v].meat[r, cc]) && continue
                k += 1; f += isfat(masks[v].lab[r, cc, 1], masks[v].lab[r, cc, 2], masks[v].lab[r, cc, 3])
            end
            k == 0 && continue
            if z == ze || (ze - z) < zstep / 2
                f / k < 0.5 && break                                # the edge reads lean: no cap here
            end
            if f / k < 0.5
                run += 1
                run == 1 && (zlean = z)
                run >= need && break
            else
                run = 0
            end
        end
        (isnan(ze) || run < need) && continue
        bands[q] = ze - zlean - zstep / 2
    end
    b = sort(filter(!isnan, bands))
    length(b) < 10 && return NaN, length(b), b
    n = length(b); lo = n ÷ 4 + 1; hi = n - n ÷ 4
    return mean(b[lo:hi]), n, b
end

# ---- the labels ---------------------------------------------------------------------------------------------------

"""
    label_voxels(g, occ, S, frac, cap_mm)

Lean everywhere, fat within `cap_mm` of a fat surface voxel when that surface is also the nearest surface (the
unseen surface voxels take their nearest seen voxel's label).
"""
function label_voxels(g::VoxelGrid, occ, S, frac, cap_mm)
    seen = findall(!isnan, frac)
    fatsurf = falses(size(occ)); anysurf = falses(size(occ))
    for s in eachindex(S)
        anysurf[S[s]] = true
    end
    for s in seen
        frac[s] >= 0.5 && (fatsurf[S[s]] = true)
    end
    df = edt(fatsurf) .* g.h; da = edt(anysurf) .* g.h
    L = fill(EMPTY, size(occ))
    for I in findall(occ)
        L[I] = (df[I] <= cap_mm && df[I] <= da[I] + 0.5g.h) ? FAT : LEAN
    end
    return L
end

# the shape's principal frame: its centroid and axes (longest first), the third flipped to point up
function principal_frame(g::VoxelGrid, occ)
    P = reduce(hcat, (collect(cell_centre(g, Tuple(I)...)) for I in findall(occ)))
    c = vec(mean(P; dims=2))
    U = svd(P .- c).U
    U[3, 3] < 0 && (U[:, 3] .= -U[:, 3])
    det(U) < 0 && (U[:, 2] .= -U[:, 2])
    return c, U
end

"""
    place_bone!(L, g, occ, profile; bone_side=:down)

The meat's bone template in its principal frame: for ribs, rods across the rack (25 mm apart, 11 mm across) at a
third of the thickness from the bone side; for a pork butt, a blade plate (100 × 60 × 5 mm) at the centroid, upright
along the long axis. A template voxel outside the meat is skipped.
"""
function place_bone!(L, g::VoxelGrid, occ, p::MeatProfile; bone_side=:down)
    p.bone === :none && return L
    c, U = principal_frame(g, occ)
    pts = [(I, collect(cell_centre(g, Tuple(I)...))) for I in findall(occ)]
    loc = [U' * (X - c) for (_, X) in pts]
    if p.bone === :rib_rods
        a = getindex.(loc, 1); zc = [X[3] for (_, X) in pts]
        zlo, zhi = extrema(zc)
        zb = bone_side === :down ? zlo + (zhi - zlo) / 3 : zhi - (zhi - zlo) / 3
        amin, amax = extrema(a)
        for (n, (I, X)) in enumerate(pts)
            t = a[n] - amin
            (t < 12 || t > amax - amin - 12) && continue
            r = rem(t - 12.5, 25.0, RoundNearest)
            hypot(r, X[3] - zb) <= 5.5 && (L[I] = BONE)
        end
    elseif p.bone === :blade_plate
        for (n, (I, _)) in enumerate(pts)
            q = loc[n]
            abs(q[1]) <= 50 && abs(q[2]) <= 2.5 && abs(q[3]) <= 30 && (L[I] = BONE)
        end
    end
    return L
end
