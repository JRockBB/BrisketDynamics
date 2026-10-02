# A cut of any shape: the labels of a reconstruction (tools/cutvision; 0 empty, 1 lean, 2 fat, 3 bone on cells of
# side h, z up from the board, the first layer on it) coarsened to the simulation's cells and laid out as a GridMesh.
# Every node is an occupied cell (its volume the block's occupied fraction), every pair of face neighbours an edge,
# every face against empty space exposed (the board's side too: the grate), its area the smooth surface's through
# the fine cells there (a staircase of cell faces overstates a sloped surface's area by up to √3), its inward line
# the cells behind it. The materials are the meat's preset (lean), the fat cap's composition, and bone (thermal
# only, as the slab's).

const MEAT_PRESETS = (brisket=brisket, pork_butt=pork_butt, ribs=ribs, chuck=chuck)

"""
    meat_preset(meat::Symbol)

The `(θ, cfg)` preset of `:brisket`, `:pork_butt`, `:ribs` or `:chuck`.
"""
function meat_preset(meat::Symbol)
    hasproperty(MEAT_PRESETS, meat) || throw(ArgumentError("unknown meat $meat; expected one of $(keys(MEAT_PRESETS))"))
    return getproperty(MEAT_PRESETS, meat)()
end

# ---- distances and smoothing on a grid ------------------------------------------------------------------------------

const _VBIG = 1e20

# the squared distance transform of one line (Felzenszwalb–Huttenlocher), f the squared distances so far
function _vedt1!(f, out, v, z)
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

# the Euclidean distance [cells] from every cell to the nearest `src` cell
function _vedt(src::AbstractArray{Bool,3})
    D = map(b -> b ? 0.0 : _VBIG, src)
    for d in 1:3
        n = size(D, d)
        f = zeros(n); out = zeros(n); v = zeros(Int, n); z = zeros(n + 1)
        for I in CartesianIndices(Base.setindex(size(D), 1, d))
            for q in 1:n
                f[q] = D[Base.setindex(Tuple(I), q, d)...]
            end
            _vedt1!(f, out, v, z)
            for q in 1:n
                D[Base.setindex(Tuple(I), q, d)...] = out[q]
            end
        end
    end
    return map(x -> x >= _VBIG / 2 ? Inf : sqrt(x), D)
end

# a separable Gaussian blur (σ in cells); below the first layer the field continues as the first layer (the board
# is not a surface), beyond the other faces it is zero
function _vblur(A::AbstractArray{Float64,3}, σ)
    r = ceil(Int, 3σ); w = [exp(-x^2 / (2σ^2)) for x in -r:r]; w ./= sum(w)
    for d in 1:3
        B = zeros(size(A)); n = size(A, d)
        for I in CartesianIndices(A)
            s = 0.0
            for (t, wt) in zip(-r:r, w)
                q = I[d] + t
                if 1 <= q <= n
                    s += wt * A[Base.setindex(Tuple(I), q, d)...]
                elseif d == 3 && q < 1
                    s += wt * A[Base.setindex(Tuple(I), 1, d)...]
                end
            end
            B[I] = s
        end
        A = B
    end
    return A
end

# ---- the smooth surface's area ----------------------------------------------------------------------------------

# the six tetrahedra of a cube around its main diagonal (corners numbered x fastest)
const _TETS = ((1, 2, 4, 8), (1, 4, 3, 8), (1, 3, 7, 8), (1, 7, 5, 8), (1, 5, 6, 8), (1, 6, 2, 8))
const _CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))

_cross(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
_tri_area(a, b, c) = 0.5 * sqrt(sum(abs2, _cross(b .- a, c .- a)))
_lerp(p, q, vp, vq) = p .+ (vp / (vp - vq)) .* (q .- p)

"""
    _iso_triangles(F, h)

The zero level set of `F` (values at cell centres, cell (i, j, k) centred at h·(i − ½, j − ½, k − ½)) by marching
tetrahedra: each triangle's area and centroid.
"""
function _iso_triangles(F::AbstractArray{Float64,3}, h)
    areas = Float64[]; cents = NTuple{3,Float64}[]
    nx, ny, nz = size(F)
    for k in 1:(nz - 1), j in 1:(ny - 1), i in 1:(nx - 1)
        vals = ntuple(c -> F[i + _CORNERS[c][1], j + _CORNERS[c][2], k + _CORNERS[c][3]], 8)
        (all(>(0), vals) || all(<=(0), vals)) && continue
        pts = ntuple(c -> h .* (i - 0.5 + _CORNERS[c][1], j - 0.5 + _CORNERS[c][2], k - 0.5 + _CORNERS[c][3]), 8)
        for t in _TETS
            v = ntuple(m -> vals[t[m]], 4); p = ntuple(m -> pts[t[m]], 4)
            pos = [m for m in 1:4 if v[m] > 0]; neg = [m for m in 1:4 if v[m] <= 0]
            tris = if length(pos) == 1 || length(pos) == 3
                a = length(pos) == 1 ? pos[1] : neg[1]
                others = length(pos) == 1 ? neg : pos
                [Tuple(_lerp(p[a], p[o], v[a], v[o]) for o in others)]
            elseif length(pos) == 2
                a, b = pos; c, d = neg
                q1 = _lerp(p[a], p[c], v[a], v[c]); q2 = _lerp(p[a], p[d], v[a], v[d])
                q3 = _lerp(p[b], p[d], v[b], v[d]); q4 = _lerp(p[b], p[c], v[b], v[c])
                [(q1, q2, q3), (q1, q3, q4)]
            else
                continue
            end
            for (x, y, z) in tris
                push!(areas, _tri_area(x, y, z)); push!(cents, (x .+ y .+ z) ./ 3)
            end
        end
    end
    return areas, cents
end

# ---- the mesh ---------------------------------------------------------------------------------------------------

const _DIRS6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

"""
    _voxel_mesh(labels, h, k; keep=0.5)

The GridMesh of `labels` (cells of side `h` [m]) on cells of side H = k·h: a node per block of k³ cells at least
`keep` occupied (its volume the occupied fraction of H³, all volumes then scaled so the total is the labels' own),
its material the block's majority (ties lean). Returns the mesh and each node's label.
"""
function _voxel_mesh(labels::AbstractArray{<:Integer,3}, h::Float64, k::Int; keep=0.5)
    k >= 1 || throw(ArgumentError("coarsen must be at least 1"))
    any(!=(0), labels) || throw(ArgumentError("the labels hold no meat"))
    fn = size(labels); cn = cld.(fn, k); H = k * h
    # the blocks
    frac = zeros(cn); code = zeros(Int8, cn)
    for J in CartesianIndices(cn)
        cnt = zeros(Int, 3); occ = 0
        for I in CartesianIndices(ntuple(d -> ((J[d] - 1) * k + 1):min(J[d] * k, fn[d]), 3))
            l = labels[I]; l == 0 && continue
            occ += 1; cnt[clamp(l, 1, 3)] += 1
        end
        frac[J] = occ / k^3
        occ > 0 && (code[J] = Int8(argmax(cnt)))
    end
    occC = frac .>= keep
    any(occC) || throw(ArgumentError("no block is $(keep) occupied: coarsen less"))
    id = zeros(Int, cn); nodes = findall(occC)
    for (n, J) in enumerate(nodes)
        id[J] = n
    end
    N = length(nodes)
    vol = [frac[J] * H^3 for J in nodes]
    vol .*= count(!=(0), labels) * h^3 / sum(vol)
    coords = zeros(N, 3)
    for (n, J) in enumerate(nodes)
        coords[n, :] = [(J[1] - 0.5) * H, (J[2] - 0.5) * H, (J[3] - 0.5) * H]
    end
    edges = Tuple{Int,Int,Float64}[]
    for (n, J) in enumerate(nodes), d in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        K = J + CartesianIndex(d)
        checkbounds(Bool, occC, K) && occC[K] && push!(edges, (n, id[K], H))
    end
    # the exposed faces: (node, outward direction); the board's under the first layer
    faces = Tuple{Int,Int}[]
    for (n, J) in enumerate(nodes), (di, d) in enumerate(_DIRS6)
        K = J + CartesianIndex(d)
        (!checkbounds(Bool, occC, K) || !occC[K]) && push!(faces, (n, di))
    end
    # each face's area: the board's faces their occupied fine cells' bottoms; the others the smooth surface's
    # triangles through the fine cells, each to the nearest non-board face of its block or a neighbour's
    area = zeros(length(faces))
    fc = [Tuple(coords[n, :]) .+ (H / 2) .* _DIRS6[di] for (n, di) in faces]
    onboard = [di == 6 && nodes[n][3] == 1 for (n, di) in faces]
    for (f, (n, _)) in enumerate(faces)
        onboard[f] || continue
        J = nodes[n]
        area[f] = h^2 * count(I -> labels[I[1], I[2], 1] != 0,
                              CartesianIndices((((J[1] - 1) * k + 1):min(J[1] * k, fn[1]), ((J[2] - 1) * k + 1):min(J[2] * k, fn[2]))))
    end
    near = Dict{NTuple{3,Int},Vector{Int}}()                         # block → its non-board faces
    for (f, (n, _)) in enumerate(faces)
        onboard[f] && continue
        push!(get!(near, Tuple(nodes[n]), Int[]), f)
    end
    pad = 2
    Fz = zeros(fn .+ (2pad, 2pad, 2pad))
    for I in CartesianIndices(labels)
        labels[I] != 0 && (Fz[I + CartesianIndex(pad, pad, pad)] = 1.0)
    end
    for i in axes(Fz, 1), j in axes(Fz, 2), z in 1:pad                # the board: the first layer continues below
        Fz[i, j, z] = Fz[i, j, pad + 1]
    end
    tri_a, tri_c = _iso_triangles(_vblur(Fz, 1.0) .- 0.5, h)
    lost = 0.0
    for (a, c) in zip(tri_a, tri_c)
        x = c .- pad * h                                              # back to the labels' frame
        J = ntuple(d -> clamp(floor(Int, x[d] / H) + 1, 1, cn[d]), 3)
        best = 0; bd = Inf
        for di in -1:1, dj in -1:1, dk in -1:1
            for f in get(near, (J[1] + di, J[2] + dj, J[3] + dk), Int[])
                dd = sum(abs2, x .- fc[f]); dd < bd && ((bd, best) = (dd, f))
            end
        end
        best == 0 ? (lost += a) : (area[best] += a)
    end
    keepf = area .> 1e-6 * H^2
    faces = faces[keepf]; area = area[keepf]
    boundary = [(n, area[f]) for (f, (n, _)) in enumerate(faces)]
    # each exposed face's inward line: the nodes behind it, at depths from the surface
    inward = Vector{Int}[]; inward_d = Vector{Float64}[]
    for (n, di) in faces
        d = _DIRS6[di]; line = Int[]; J = nodes[n]
        while checkbounds(Bool, occC, J) && occC[J]
            push!(line, id[J]); J = J - CartesianIndex(d)
        end
        push!(inward, line); push!(inward_d, [H / 2 + (m - 1) * H for m in eachindex(line)])
    end
    # each node's depth below the nearest exposed face (the board's included: padded with empty cells all round)
    E = trues(cn .+ 2)
    for J in nodes
        E[J + CartesianIndex(1, 1, 1)] = false
    end
    D = _vedt(E)
    depth = [max(D[J + CartesianIndex(1, 1, 1)] - 0.5, 0.5) * H for J in nodes]
    # the core: of the deepest cells (they tie across a slab's middle), the one nearest their centroid
    deepest = findall(d -> d >= maximum(depth) - 1e-9 * H, depth)
    cen = vec(sum(coords[deepest, :]; dims=1)) ./ length(deepest)
    core = deepest[argmin([sum(abs2, coords[n, :] .- cen) for n in deepest])]
    # the core line: steepest descent of the depth from the core to an exposed node
    exposed = falses(N)
    for (n, _) in faces
        exposed[n] = true
    end
    line = [core]
    while !exposed[line[end]]
        J = nodes[line[end]]; nxt = 0; dn = depth[line[end]]
        for d in _DIRS6
            K = J + CartesianIndex(d)
            (checkbounds(Bool, occC, K) && occC[K]) || continue
            depth[id[K]] < dn && ((dn, nxt) = (depth[id[K]], id[K]))
        end
        nxt == 0 && break
        push!(line, nxt)
    end
    mesh = GridMesh(N, vol, edges, boundary, coords, core, line, sum(last.(boundary)), cn, (cn[1] * H, cn[2] * H, cn[3] * H),
                    reshape(depth, N, 1), depth, inward, inward_d)
    return mesh, [code[J] for J in nodes], lost
end

"""
    VoxelPiece(id, labels, h; meat=:brisket, coarsen=2, cfg=nothing)

A cut of any shape from a reconstruction's labels (0 empty, 1 lean, 2 fat, 3 bone; cells of side `h` [m], z up from
the board, the first layer on it), on cells `coarsen` times larger (`_voxel_mesh`). The lean is the `meat`'s preset
(`brisket`, `pork_butt`, `ribs`, `chuck`; `cfg` overrides it), the fat the cap's composition (`FAT_CAP_F0`,
`FAT_CAP_W0`), the bone thermal-only bone. The whole cut is solved: the piece's area multiplier is 1.
"""
function VoxelPiece(id::Symbol, labels::AbstractArray{<:Integer,3}, h::Real; meat::Symbol=:brisket, coarsen::Int=2,
                    cfg::Union{Nothing,MeatConfig}=nothing)
    c = with_config(something(cfg, meat_preset(meat).cfg); A_meat=1.0)
    mesh, codes, _ = _voxel_mesh(labels, Float64(h), coarsen)
    mats = [c, with_config(c; F0=FAT_CAP_F0, W0=FAT_CAP_W0)]
    mat = [l == 2 ? 2 : 1 for l in codes]
    bone = [l == 3 ? 1.0 : 0.0 for l in codes]
    return GridPiece(id, c, mesh, :voxel3d, mat, mats, bone)
end
