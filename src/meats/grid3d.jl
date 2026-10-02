# A piece on a structured finite-volume grid (docs/multi-meat-inference-plan.md, 3-D and reduced-order
# models): the full-order model of a cut in three dimensions (`Brick3D`, a box on a Cartesian node-centred
# grid with any subset of its six faces exposed) and the axisymmetric cylinder of slab2d.jl adapted to the
# `PieceModel` interface (`Cylinder2D`, an r–z grid, the rim and optionally the ends exposed), both ONE model
# (`GridPiece`) over a `GridMesh`: nodes with volumes, edges with a geometric conductance (face area over
# distance) and exposed boundary nodes with their face areas. The cell physics is the slab's (meats/common.jl:
# Choi–Okos conductivity and capacity from the local composition, Oswin water activity, the two-sided
# evaporation, the wrap and pull blends, the expulsion sink, fat rendering, collagen conversion, Maillard bark,
# smoke deposition and the ring on every exposed node) as a method-of-lines ODE the system's FBDF integrates beside the
# pit, so a grid piece couples through the same exchange port and the same chamber-humidity closure as a slab.
#
# The slab's depth physics on the mesh: the material-coordinate shrink per node (each edge's distance stretched by
# its nodes' composition, the exposed-area factor on the surface water flux and the pit's share), the dry-brine
# salt field (the union of the erfc profiles from every exposed face, diffusing, holding water and throttling the
# surface's vapor pressure), a bone at the core (on the depth below the nearest exposed face) and the smoke-ring
# front per exposed node (along the node's inward line). The `A_meat` parameter is read as a multiplier on the
# mesh's exposed area. The one-dimensional limit (a brick of one transverse node with one exposed face)
# reproduces the slab's arithmetic term for term, which test/test_grid3d.jl gates at the default configuration.

using LinearAlgebra: dot

"""
    GridMesh

Nodes with volumes `vol` [m³], edges `(i, j, g)` with `g` the face area over the node distance [m] (the flux
between i and j is `g · k_face · (T_j − T_i)`), exposed boundary entries `(node, area)` [m²] (a node on two
exposed faces appears twice), the node coordinates, the core node (farthest from every exposed face), the
through-thickness line (the nodes from the core to the nearest exposed face, for a reduced-order projection)
and the total exposed area. For the depth physics: `face_dist` (N × the exposed faces) each node's distance to
every exposed face (the brine profile is their union), `depth` the nearest of them (the bone's coordinate), and
per boundary entry its `inward` line (the nodes from it along the face's inward normal) with the nodes' depths
along it `inward_d` (the smoke-ring front reads the temperature at its depth there).
"""
struct GridMesh
    N::Int
    vol::Vector{Float64}
    edges::Vector{Tuple{Int,Int,Float64}}
    boundary::Vector{Tuple{Int,Float64}}
    coords::Matrix{Float64}
    core::Int
    line::Vector{Int}
    area::Float64
    dims::Tuple{Int,Int,Int}
    extents::NTuple{3,Float64}
    face_dist::Matrix{Float64}
    depth::Vector{Float64}
    inward::Vector{Vector{Int}}
    inward_d::Vector{Vector{Float64}}
end

# node-centred tensor grid on a box: nodes at i·Δ (i = 0 … n−1), half cells at the ends, face areas the product
# of the transverse cell widths
function _brick_mesh(Lx, Ly, Lz, nx, ny, nz, exposed::NTuple{6,Bool})
    all(isfinite, (Lx, Ly, Lz)) && all(>(0), (Lx, Ly, Lz)) ||
        throw(ArgumentError("brick dimensions must be finite and positive"))
    all(>=(1), (nx, ny, nz)) && any(>=(2), (nx, ny, nz)) ||
        throw(ArgumentError("a brick grid needs positive node counts and at least one resolved direction"))
    ns = (nx, ny, nz); Ls = (Lx, Ly, Lz)
    Δ = ntuple(d -> ns[d] > 1 ? Ls[d] / (ns[d] - 1) : Ls[d], 3)
    w(d, i) = ns[d] == 1 ? Ls[d] : ((i == 1 || i == ns[d]) ? Δ[d] / 2 : Δ[d])   # the cell width of node i along d
    idx(i, j, k) = i + (j - 1) * nx + (k - 1) * nx * ny
    N = nx * ny * nz
    vol = zeros(N); coords = zeros(N, 3)
    for k in 1:nz, j in 1:ny, i in 1:nx
        n = idx(i, j, k)
        vol[n] = w(1, i) * w(2, j) * w(3, k)
        coords[n, :] = [(i - 1) * Δ[1], (j - 1) * Δ[2], (k - 1) * Δ[3]]
    end
    edges = Tuple{Int,Int,Float64}[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        i < nx && push!(edges, (idx(i, j, k), idx(i + 1, j, k), w(2, j) * w(3, k) / Δ[1]))
        j < ny && push!(edges, (idx(i, j, k), idx(i, j + 1, k), w(1, i) * w(3, k) / Δ[2]))
        k < nz && push!(edges, (idx(i, j, k), idx(i, j, k + 1), w(1, i) * w(2, j) / Δ[3]))
    end
    boundary = Tuple{Int,Float64}[]
    inward = Vector{Int}[]; inward_d = Vector{Float64}[]
    function face!(node, area, line, Δl)
        push!(boundary, (node, area)); push!(inward, line); push!(inward_d, Δl .* (0:(length(line) - 1)))
    end
    for k in 1:nz, j in 1:ny
        exposed[1] && face!(idx(1, j, k), w(2, j) * w(3, k), [idx(i, j, k) for i in 1:nx], Δ[1])
        exposed[2] && face!(idx(nx, j, k), w(2, j) * w(3, k), [idx(i, j, k) for i in nx:-1:1], Δ[1])
    end
    for k in 1:nz, i in 1:nx
        exposed[3] && face!(idx(i, 1, k), w(1, i) * w(3, k), [idx(i, j, k) for j in 1:ny], Δ[2])
        exposed[4] && face!(idx(i, ny, k), w(1, i) * w(3, k), [idx(i, j, k) for j in ny:-1:1], Δ[2])
    end
    for j in 1:ny, i in 1:nx
        exposed[5] && face!(idx(i, j, 1), w(1, i) * w(2, j), [idx(i, j, k) for k in 1:nz], Δ[3])
        exposed[6] && face!(idx(i, j, nz), w(1, i) * w(2, j), [idx(i, j, k) for k in nz:-1:1], Δ[3])
    end
    isempty(boundary) && throw(ArgumentError("a brick needs at least one exposed face"))
    # the core: farthest from every exposed face; the line: from the core toward the nearest exposed face along that axis
    nf = count(exposed)
    face_dist = zeros(N, nf)
    for n in 1:N
        x, y, z = coords[n, :]
        ds = Float64[]
        exposed[1] && push!(ds, x); exposed[2] && push!(ds, Lx - x)
        exposed[3] && push!(ds, y); exposed[4] && push!(ds, Ly - y)
        exposed[5] && push!(ds, z); exposed[6] && push!(ds, Lz - z)
        face_dist[n, :] = ds
    end
    dist = vec(minimum(face_dist; dims=2))
    core = argmax(dist)
    ci = (core - 1) % nx + 1; cj = ((core - 1) ÷ nx) % ny + 1; ck = (core - 1) ÷ (nx * ny) + 1
    x, y, z = coords[core, :]
    cands = Tuple{Float64,Vector{Int}}[]
    exposed[1] && push!(cands, (x, [idx(i, cj, ck) for i in ci:-1:1]))
    exposed[2] && push!(cands, (Lx - x, [idx(i, cj, ck) for i in ci:nx]))
    exposed[3] && push!(cands, (y, [idx(ci, j, ck) for j in cj:-1:1]))
    exposed[4] && push!(cands, (Ly - y, [idx(ci, j, ck) for j in cj:ny]))
    exposed[5] && push!(cands, (z, [idx(ci, cj, k) for k in ck:-1:1]))
    exposed[6] && push!(cands, (Lz - z, [idx(ci, cj, k) for k in ck:nz]))
    line = cands[argmin(first.(cands))][2]
    return GridMesh(N, vol, edges, boundary, coords, core, line, sum(last.(boundary)), (nx, ny, nz),
                    (Float64(Lx), Float64(Ly), Float64(Lz)), face_dist, dist, inward, inward_d)
end

# the axisymmetric r–z grid of slab2d.jl: node i on the axis (r = 0) to the rim, half cells at the axis, the rim and the
# ends; radial faces ∝ r_face, volumes ∝ ∫ r dr, exactly the ADI engine's rF and rV
function _cylinder_mesh(R, Hz, nr, nz, exposed_ends::Bool)
    isfinite(R) && R > 0 && isfinite(Hz) && Hz > 0 ||
        throw(ArgumentError("cylinder radius and height must be finite and positive"))
    nr >= 2 && nz >= 2 || throw(ArgumentError("a cylinder grid needs at least two radial and two axial nodes"))
    Δr = R / (nr - 1); Δz = Hz / (nz - 1)
    rF = [(i - 0.5) * Δr for i in 1:(nr - 1)]
    rV = [i == 1 ? Δr^2 / 8 : i == nr ? ((nr - 1)^2 - (nr - 1.5)^2) * Δr^2 / 2 : ((i - 0.5)^2 - (i - 1.5)^2) * Δr^2 / 2 for i in 1:nr]
    zw = [j == 1 || j == nz ? Δz / 2 : Δz for j in 1:nz]
    idx(i, j) = i + (j - 1) * nr
    N = nr * nz
    vol = zeros(N); coords = zeros(N, 3)
    for j in 1:nz, i in 1:nr
        vol[idx(i, j)] = 2π * rV[i] * zw[j]
        coords[idx(i, j), :] = [(i - 1) * Δr, (j - 1) * Δz, 0.0]
    end
    edges = Tuple{Int,Int,Float64}[]
    for j in 1:nz, i in 1:nr
        i < nr && push!(edges, (idx(i, j), idx(i + 1, j), 2π * rF[i] * zw[j] / Δr))
        j < nz && push!(edges, (idx(i, j), idx(i, j + 1), 2π * rV[i] / Δz))
    end
    boundary = Tuple{Int,Float64}[]
    inward = Vector{Int}[]; inward_d = Vector{Float64}[]
    for j in 1:nz
        push!(boundary, (idx(nr, j), 2π * R * zw[j])); push!(inward, [idx(i, j) for i in nr:-1:1]); push!(inward_d, Δr .* (0:(nr - 1)))
    end
    if exposed_ends
        for i in 1:nr
            push!(boundary, (idx(i, 1), 2π * rV[i])); push!(inward, [idx(i, j) for j in 1:nz]); push!(inward_d, Δz .* (0:(nz - 1)))
            push!(boundary, (idx(i, nz), 2π * rV[i])); push!(inward, [idx(i, j) for j in nz:-1:1]); push!(inward_d, Δz .* (0:(nz - 1)))
        end
    end
    # each node's distance to the rim and, when exposed, to both ends
    face_dist = zeros(N, exposed_ends ? 3 : 1)
    for j in 1:nz, i in 1:nr
        n = idx(i, j); r = (i - 1) * Δr; z = (j - 1) * Δz
        face_dist[n, 1] = R - r
        exposed_ends && (face_dist[n, 2] = z; face_dist[n, 3] = Hz - z)
    end
    core = idx(1, div(nz, 2) + 1)
    line = [idx(i, div(nz, 2) + 1) for i in 1:nr]
    return GridMesh(N, vol, edges, boundary, coords, core, line, sum(last.(boundary)), (nr, nz, 1),
                    (Float64(R), Float64(Hz), 0.0), face_dist, vec(minimum(face_dist; dims=2)), inward, inward_d)
end


# A trimmed brisket flat: a slab tapering in thickness and width along its length, on a mapped node-centred grid. The
# computational cube (ξ, η, ζ) ∈ [0, 1]³ maps to the length s = ξ·Len (s = 0 the thick end, toward the point), the
# width y = η·b(s) from the centre plane (b the half-width, linear from the thick end's to the thin end's) and the
# height z above the grate (H the thickness, linear likewise): a flat bottom on the grate, the top sloping. A fat cap
# of constant thickness `cap` is resolved by its own `ncap` node layers: the lean below spaced (H − cap)/(nt − ncap −
# ½), the cap spaced cap/(ncap − ½) down from the top, the dual face between the two layers on the cap's underside
# (each a half cell of its own spacing from it: every node is lean or cap whole, the same layers along the length); `ncap = 0` spaces the thickness
# uniformly, z = ζ·H. Half the cut is solved, across its width: the centre plane y = 0 is a symmetry plane; the grate
# side, the top, the outer side and both ends are exposed. A node's volume is Len·b·Δz at its position times its
# computational cell along the length and the width (half cells at the faces), Δz its dual cell's height; an edge's
# conductance is its dual face's area over the distance between its nodes (two-point flux on a mildly skewed grid:
# the taper's slopes are a tenth, the non-orthogonality error their square); an exposed face's area is the mapped
# surface element's.
function _flat_mesh(Len, w0, w1, h0, h1, nl, nw, nt; cap=0.0, ncap::Int=0)
    all(isfinite, (Len, w0, w1, h0, h1)) && all(>(0), (Len, w0, w1, h0, h1)) ||
        throw(ArgumentError("a flat's length, widths and thicknesses must be finite and positive"))
    nl >= 2 && nw >= 2 && nt >= 2 || throw(ArgumentError("a flat's grid needs at least two nodes along every direction"))
    ncap == 0 || (cap > 0 && cap < min(h0, h1) && 1 <= ncap <= nt - 2) ||
        throw(ArgumentError("a resolved fat cap needs 0 < cap < the thinnest thickness and 1 ≤ ncap ≤ nt − 2"))
    b(s) = (w0 + (w1 - w0) * s / Len) / 2
    H(s) = h0 + (h1 - h0) * s / Len
    db = (w1 - w0) / (2Len); dH = (h1 - h0) / Len
    dξ, dη = 1 / (nl - 1), 1 / (nw - 1)
    cell(d, i, n) = (i == 1 || i == n) ? d / 2 : d                  # the computational cell width of node i of n
    nlean = nt - ncap
    zk(k, s) = ncap == 0 ? (k - 1) * H(s) / (nt - 1) :
               k <= nlean ? (k - 1) * (H(s) - cap) / (nlean - 0.5) : H(s) - (nt - k) * cap / (ncap - 0.5)
    zf(k, s) = k == 0 ? 0.0 : k == nt ? H(s) : (ncap > 0 && k == nlean) ? H(s) - cap : (zk(k, s) + zk(k + 1, s)) / 2
    Δz(k, s) = zf(k, s) - zf(k - 1, s)                                # the dual cell between faces k − 1 and k
    idx(i, j, k) = i + (j - 1) * nl + (k - 1) * nl * nw
    N = nl * nw * nt
    s_of(i) = (i - 1) * dξ * Len
    pos(i, j, k) = (s = s_of(i); (s, (j - 1) * dη * b(s), zk(k, s)))
    vol = zeros(N); coords = zeros(N, 3)
    for k in 1:nt, j in 1:nw, i in 1:nl
        n = idx(i, j, k); s = s_of(i)
        vol[n] = Len * b(s) * Δz(k, s) * cell(dξ, i, nl) * cell(dη, j, nw)
        coords[n, :] = collect(pos(i, j, k))
    end
    dist(a, c) = sqrt(sum(abs2, (coords[a, 1] - coords[c, 1], coords[a, 2] - coords[c, 2], coords[a, 3] - coords[c, 3])))
    edges = Tuple{Int,Int,Float64}[]
    for k in 1:nt, j in 1:nw, i in 1:nl
        n = idx(i, j, k); s = s_of(i)
        if i < nl                                                     # along the length: the face at the midpoint
            m = idx(i + 1, j, k); sm = s + dξ * Len / 2
            push!(edges, (n, m, b(sm) * Δz(k, sm) * cell(dη, j, nw) / dist(n, m)))
        end
        if j < nw                                                     # across the width
            m = idx(i, j + 1, k)
            push!(edges, (n, m, Len * Δz(k, s) * cell(dξ, i, nl) / dist(n, m)))
        end
        if k < nt                                                     # through the thickness
            m = idx(i, j, k + 1)
            push!(edges, (n, m, Len * b(s) * cell(dξ, i, nl) * cell(dη, j, nw) / dist(n, m)))
        end
    end
    boundary = Tuple{Int,Float64}[]; inward = Vector{Int}[]; inward_d = Vector{Float64}[]
    face!(node, area, line, dl) = (push!(boundary, (node, area)); push!(inward, line); push!(inward_d, dl))
    for j in 1:nw, i in 1:nl                                          # the top (sloping) and the grate side
        s = s_of(i); a = Len * b(s) * cell(dξ, i, nl) * cell(dη, j, nw)
        face!(idx(i, j, nt), a * sqrt(1 + dH^2), [idx(i, j, k) for k in nt:-1:1], [H(s) - zk(k, s) for k in nt:-1:1])
        face!(idx(i, j, 1), a, [idx(i, j, k) for k in 1:nt], [zk(k, s) for k in 1:nt])
    end
    for k in 1:nt, i in 1:nl                                          # the outer side (sloping in plan)
        s = s_of(i)
        face!(idx(i, nw, k), Len * Δz(k, s) * sqrt(1 + db^2) * cell(dξ, i, nl),
              [idx(i, j, k) for j in nw:-1:1], dη * b(s) .* (0:(nw - 1)))
    end
    for k in 1:nt, j in 1:nw                                          # the thick end and the thin end
        face!(idx(1, j, k), b(0.0) * Δz(k, 0.0) * cell(dη, j, nw), [idx(i, j, k) for i in 1:nl], dξ * Len .* (0:(nl - 1)))
        face!(idx(nl, j, k), b(Len) * Δz(k, Len) * cell(dη, j, nw), [idx(i, j, k) for i in nl:-1:1], dξ * Len .* (0:(nl - 1)))
    end
    # every node's distance to each exposed face (the ends along the length, the side across the width, the grate and
    # the top vertically: the taper's slopes make these the normal distances to within their square)
    face_dist = zeros(N, 5)
    for n in 1:N
        s, y, z = coords[n, 1], coords[n, 2], coords[n, 3]
        face_dist[n, :] = [s, Len - s, b(s) - y, z, H(s) - z]
    end
    depth = vec(minimum(face_dist; dims=2))
    core = argmax(depth)
    ci = (core - 1) % nl + 1; cj = ((core - 1) ÷ nl) % nw + 1; ck = (core - 1) ÷ (nl * nw) + 1
    # the core line: from the core to the nearest exposed face along that face's grid line
    cands = [(face_dist[core, 5], [idx(ci, cj, k) for k in ck:nt]), (face_dist[core, 4], [idx(ci, cj, k) for k in ck:-1:1]),
             (face_dist[core, 3], [idx(ci, j, ck) for j in cj:nw]),
             (face_dist[core, 1], [idx(i, cj, ck) for i in ci:-1:1]), (face_dist[core, 2], [idx(i, cj, ck) for i in ci:nl])]
    line = cands[argmin(first.(cands))][2]
    return GridMesh(N, vol, edges, boundary, coords, core, line, sum(last.(boundary)), (nl, nw, nt),
                    (Float64(Len), Float64(w0 / 2), Float64(h0)), face_dist, depth, inward, inward_d)
end

"""
    GridPiece

A piece on a `GridMesh` with the slab's constitutive constants (`cfg`; its `nx` is unused) and a material per node:
`mats` the configurations (the first the cut's own, `cfg`), `mat` each node's index into them (a fat cap is a
second material: its own initial water and fat, every property and rate read from it). `Brick3D`, `Cylinder2D` and
`BrisketFlat3D` build it; one material is the slab's arithmetic exactly.
"""
struct GridPiece <: PieceModel
    id::Symbol
    cfg::MeatConfig
    mesh::GridMesh
    kind::Symbol
    mat::Vector{Int}
    mats::Vector{MeatConfig}
end
GridPiece(id::Symbol, cfg::MeatConfig, mesh::GridMesh, kind::Symbol) = GridPiece(id, cfg, mesh, kind, ones(Int, mesh.N), [cfg])
piece_parameter_indices(::GridPiece) = (1, 2, 3, 5, 6, 7, 8) # geometry fixes L; carrying it as a latent is structurally singular
piece_restart_fields(::GridPiece) = (:T0f, :W0f, :C0f, :F0f, :Ssalt0f, :B0f, :S0f, :R0f)
piece_observable_schema(::GridPiece) = (:temperature, :temperature_location, :surface_temperature, :mass, :bark)
"""
    Brick3D(id; Lx, Ly, Lz, nx, ny, nz, exposed=(false, true, true, true, true, true), cfg=MeatConfig())

A box [0, Lx] × [0, Ly] × [0, Lz] on an nx × ny × nz node-centred grid, `exposed` its faces (x low, x high, y low,
y high, z low, z high) open to the pit; an unexposed face is insulated (a symmetry plane, a rack). The default
leaves the x-low face as the symmetry plane of a half-thickness slab.
"""
Brick3D(id::Symbol; Lx::Real, Ly::Real, Lz::Real, nx::Int, ny::Int, nz::Int, exposed::NTuple{6,Bool}=(false, true, true, true, true, true),
        cfg::MeatConfig=MeatConfig()) = GridPiece(id, with_config(cfg; A_meat=1.0), _brick_mesh(Float64(Lx), Float64(Ly), Float64(Lz), nx, ny, nz, exposed), :brick3d)
"""
    Cylinder2D(id; R, Hz, nr, nz, exposed_ends=true, cfg=MeatConfig())

The axisymmetric finite cylinder of slab2d.jl (radius `R`, height `Hz`, `nr × nz` nodes, the rim and both ends
exposed; `exposed_ends = false` insulates the ends, the infinite cylinder of the slab's `geom_m = 1`).
"""
Cylinder2D(id::Symbol; R::Real, Hz::Real, nr::Int, nz::Int, exposed_ends::Bool=true, cfg::MeatConfig=MeatConfig()) =
    GridPiece(id, with_config(cfg; A_meat=1.0), _cylinder_mesh(Float64(R), Float64(Hz), nr, nz, exposed_ends), :cylinder2d)

"""
The dimensions of a typical trimmed brisket flat [m]: 33 cm long, 23 cm wide at the thick (point) end tapering to
18 cm, 5.5 cm thick tapering to 2.5 cm, a fat cap trimmed to 6 mm on top (about 3 kg). Placeholders until the cut is
measured.
"""
const BRISKET_FLAT_TYPICAL = (length=0.33, width_thick=0.23, width_thin=0.18, thickness_thick=0.055, thickness_thin=0.025,
                              cap=0.006)

"""
The fat cap's composition, dry basis: fat tissue about 80 % lipid, 15 % water, 5 % protein by mass, so 0.94 kg fat and
0.18 kg water per kg of dry matter.
"""
const FAT_CAP_F0 = 0.94
const FAT_CAP_W0 = 0.18

"""
    BrisketFlat3D(id; length, width_thick, width_thin, thickness_thick, thickness_thin, cap, nl, nw, nt, ncap, cfg)

A trimmed brisket flat (`_flat_mesh`): thickness and width tapering linearly from the thick end to the thin end, a
flat bottom on the grate, the top sloping; half the cut solved across its width (the centre plane a symmetry plane),
the grate side, the top, the outer side and both ends exposed. The top `cap` [m] is fat cap, resolved by its own
`ncap` of the `nt` node layers along the whole length (a second material, `FAT_CAP_F0` and `FAT_CAP_W0`; `cap = 0`
none). The defaults are `BRISKET_FLAT_TYPICAL`; the piece's area multiplier 2 makes the pit see the whole cut.
"""
function BrisketFlat3D(id::Symbol; length::Real=BRISKET_FLAT_TYPICAL.length, width_thick::Real=BRISKET_FLAT_TYPICAL.width_thick,
                       width_thin::Real=BRISKET_FLAT_TYPICAL.width_thin, thickness_thick::Real=BRISKET_FLAT_TYPICAL.thickness_thick,
                       thickness_thin::Real=BRISKET_FLAT_TYPICAL.thickness_thin, cap::Real=BRISKET_FLAT_TYPICAL.cap,
                       nl::Int=12, nw::Int=6, nt::Int=8, ncap::Int=2, cfg::MeatConfig=MeatConfig())
    cap >= 0 || throw(ArgumentError("the fat cap's thickness must be nonnegative"))
    nc = cap > 0 ? ncap : 0
    mesh = _flat_mesh(Float64(length), Float64(width_thick), Float64(width_thin), Float64(thickness_thick),
                      Float64(thickness_thin), nl, nw, nt; cap=Float64(cap), ncap=nc)
    c = with_config(cfg; A_meat=1.0)
    mat = [(n - 1) ÷ (nl * nw) + 1 > nt - nc ? 2 : 1 for n in 1:mesh.N]
    mats = [c, with_config(c; F0=FAT_CAP_F0, W0=FAT_CAP_W0)]
    return GridPiece(id, c, mesh, :flat3d, mat, mats)
end

"""
    piece_solve(pc::GridPiece, θ, Tpit_knots, tknots, tsave; pv=cfg.p_v_pit, airflow=cfg.g_ref, t_wrap, t_pull, mdot=0.0, t_salt=NaN, abstol, reltol)

The grid piece alone under a prescribed pit temperature (interpolated from `Tpit_knots` on the uniform
`tknots`), a fixed chamber vapor pressure and draft (the standalone full-order model, the ADI engine's
counterpart): its `piece_outputs` with the state array `X`.
"""
function piece_solve(pc::GridPiece, θ, Tpit_knots::AbstractVector, tknots::AbstractVector, tsave::AbstractVector; pv::Real=pc.cfg.p_v_pit,
                     airflow::Real=pc.cfg.g_ref, t_wrap::Real=WRAP_NEVER, t_pull::Real=PULL_NEVER, mdot::Real=0.0, t_salt::Real=NaN,
                     abstol=1e-8, reltol=1e-8)
    tk0, tkdt = float(tknots[1]), float(tknots[2] - tknots[1])
    hoisted = _piece_hoisted(pc)
    θt = ntuple(j -> θ[j], MEAT_NP)
    rhs! = function (du, u, p, t)
        _piece_rhs!(pc, du, u, t, θt, lininterp(t, tk0, tkdt, Tpit_knots), mdot, pv, airflow, t_wrap, t_pull, SPRITZ_NEVER, 3600.0, pc.cfg.m_spritz, hoisted, 1.0)
        return nothing
    end
    u0 = initial_state(pc, θ; t_salt)
    prob = ODEProblem{true,SciMLBase.FullSpecialize}(rhs!, u0, (float(tsave[1]), float(tsave[end])), nothing)
    sol = solve(prob, FBDF(); saveat=tsave, abstol=abstol, reltol=reltol)
    A = Array(sol)
    (SciMLBase.successful_retcode(sol) && size(A, 2) == length(tsave)) || throw(ErrorException("grid piece solve failed: retcode=$(sol.retcode)"))
    nt = length(tsave)
    Tpit_ts = [lininterp(float(t), tk0, tkdt, Tpit_knots) for t in tsave]
    ps = (airflow=fill(float(airflow), nt), mdot=fill(float(mdot), nt), smoke_clean=ones(nt), pv_star=fill(float(pv), nt), Tpit=Tpit_ts)
    ev = (t_wrap=float(t_wrap), t_pull=float(t_pull), t_spritz=SPRITZ_NEVER, dt_spritz=3600.0, m_spritz=pc.cfg.m_spritz)
    return merge(piece_outputs(pc, A, θ, 1.0, ps, tsave, ev), (; X=A, t=collect(float.(tsave))))
end

n_nodes(pc::GridPiece) = pc.mesh.N
n_boundary(pc::GridPiece) = length(pc.mesh.boundary)
ode_layout(pc::GridPiece) = (N = pc.mesh.N; nb = length(pc.mesh.boundary);
    BlockLayout([(:T, N, :positive), (:W, N, :positive), (:C, N, :unit), (:F, N, :positive), (:S, N, :positive),
                 (:bark_b, nb, :positive), (:bark_s, nb, :positive), (:ring, nb, :unit)]))

# hoisted once per closure: the mesh, the shrink exponents and the static bone weights per node (the slab's blend
# `logistic((bone_frac − ξ̂)/δ_bone_frac)` on the normalized depth ξ̂ = 1 − d/d_max below the nearest exposed face;
# exactly zero for the boneless default, and the slab's ξ̂ in its one-dimensional limit)
function _piece_hoisted(pc::GridPiece)
    cfg = pc.cfg; mesh = pc.mesh
    wb = cfg.bone_frac > 0 ?
         logistic.((cfg.bone_frac .- (1 .- mesh.depth ./ maximum(mesh.depth))) ./ cfg.δ_bone_frac) :
         zeros(mesh.N)
    v0n = [specific_volume(pc.mats[m].W0, pc.mats[m].F0, pc.mats[m]) for m in pc.mat]   # each node's own reference volume
    return (mesh, shrink_exponents(cfg)..., wb, pc.mats, pc.mat, v0n)
end

"""
    grid_salt_profile(mesh, cfg, t_salt)

The dry-brine initial salt field: the slab's erfc profile from EVERY exposed (rubbed) face as a smooth union,
`S_surf·(1 − Π_f (1 − erfc(d_f/(2√(D_salt t_salt)))))` over the node's distances `d_f` to the faces (exactly the
single-face profile away from edges and corners); `t_salt ≤ 0` is exact zeros (the primal compared, so a Dual
seeded at the unsalted sentinel cannot fall into the √t singularity).
"""
function grid_salt_profile(mesh::GridMesh, cfg::MeatConfig, t_salt)
    _primal(t_salt) <= 0 && return zeros(typeof(float(t_salt)), mesh.N)
    pen = 2 * sqrt(cfg.D_salt * t_salt)
    return [cfg.S_surf * (1 - prod(1 - erfc_smooth(mesh.face_dist[n, f] / pen) for f in axes(mesh.face_dist, 2)))
            for n in 1:mesh.N]
end

"""
    initial_state(pc::GridPiece, θ; ics=(;), t_salt=NaN)

Uniform fields at the config's raw state, the brine profile from every exposed face over `t_salt` (NaN: the
config's), the bark, smoke and ring at zero; `ics` (`T0f`, `W0f`, `C0f`, `F0f`, `Ssalt0f` of the node count,
`B0f`, `S0f`, `R0f` of the boundary count) replaces them for a restart.
"""
function initial_state(pc::GridPiece, θ; ics::NamedTuple=(;), t_salt::Real=NaN)
    cfg = pc.cfg; N = pc.mesh.N; nb = length(pc.mesh.boundary)
    all(k -> k in piece_restart_fields(pc), keys(ics)) ||
        throw(ArgumentError("unsupported restart field for $(pc.id); expected $(piece_restart_fields(pc))"))
    g(k, default) = haskey(ics, k) && !isempty(getfield(ics, k)) ? collect(getfield(ics, k)) : default
    salt0 = grid_salt_profile(pc.mesh, cfg, isnan(_primal(t_salt)) ? cfg.t_salt : t_salt)
    T = g(:T0f, fill(cfg.T0, N)); C = g(:C0f, zeros(N))
    W = g(:W0f, [pc.mats[m].W0 for m in pc.mat]); F = g(:F0f, [pc.mats[m].F0 for m in pc.mat])
    Sl = g(:Ssalt0f, salt0)
    B = g(:B0f, zeros(nb)); S = g(:S0f, zeros(nb)); Rg = g(:R0f, zeros(nb))
    (length(T) == N && length(W) == N && length(C) == N && length(F) == N && length(Sl) == N) ||
        throw(ArgumentError("a grid piece's fields must have $N entries"))
    (length(B) == nb && length(S) == nb && length(Rg) == nb) ||
        throw(ArgumentError("a grid piece's surface states must have $nb entries"))
    u0 = vcat(T, W, C, F, Sl, B, S, Rg)
    all(v -> isfinite(_primal(v)), u0) || throw(ArgumentError("restart state for $(pc.id) must be finite"))
    return u0
end

# the exposed-area factor (v̄/v̄0)^n_a from the volume-weighted mean specific volume against its reference (each node
# its material's): the slab's, on the mesh
function _grid_area_factor(mesh, W, F, mats, mat, v0n, n_a)
    vsum = zero(promote_type(eltype(W), eltype(F)))
    v0sum = 0.0
    @inbounds for n in 1:mesh.N
        vsum += mesh.vol[n] * specific_volume(W[n], F[n], mats[mat[n]])
        v0sum += mesh.vol[n] * v0n[n]
    end
    return (vsum / v0sum)^n_a
end

# the exchange port: the sum over the exposed nodes of area × α × a (the conductance) and area × α × a × e, α the
# exposed-area factor and e the node's salted equilibrium vapor pressure
function _piece_exchange(pc::GridPiece, u, t, θ, airflow, t_wrap, t_pull, A_meat, hoisted)
    mesh, n_s, n_a = hoisted[1], hoisted[2], hoisted[3]
    mats, mat, v0n = hoisted[6], hoisted[7], hoisted[8]
    cfg = pc.cfg; N = mesh.N
    k_m, m_conv = θ[2], θ[8]
    k_m = k_m * (airflow / cfg.g_ref)^m_conv
    k_eff = isinf(cfg.k_skin) ? k_m : 1 / (1 / k_m + 1 / cfg.k_skin)
    s = wrap_blend(t, t_wrap, cfg)
    pl = pull_blend(t, t_pull, cfg)
    W = @view u[(N + 1):2N]; F = @view u[(3N + 1):4N]; Sl = @view u[(4N + 1):5N]
    a_area = _grid_area_factor(mesh, W, F, mats, mat, v0n, n_a)
    scale = A_meat * cfg.n_meat * a_area
    cond = zero(promote_type(eltype(u), typeof(k_m), typeof(s))); num = zero(cond)
    for (n, area) in mesh.boundary
        Ts = u[n]; Ws = W[n]; mc = mats[mat[n]]
        a = (1 - cfg.η_evap * s) * k_eff / (R_V * Ts)
        f_aw = 1 / (1 + cfg.k_aw * max(Sl[n], zero(Sl[n])))
        cond += scale * area * a
        num += scale * area * a * f_aw * a_w_oswin(Ws, mc) * p_sat(Ts)
    end
    return cond, num, a_area, s, pl
end

"""
    _piece_rhs!(pc::GridPiece, du, u, ...)

The method-of-lines right-hand side, the slab's physics on the mesh. Each node carries the material-coordinate
stretch st = (v/v0)^n_shrink of its own composition (water and rendering fat); an edge's distance stretches by the
mean of its nodes', heat is per physical volume (node capacity ρc·vol·st, conductance g·k̄/st̄) and water and salt
per dry mass (conductance g·D/st̄²), the slab's interior form; the conductivity and capacity blend to bone by the
static weights. Per node the expulsion sink (raised by the salt's water-holding), fat rendering and collagen
conversion; on every exposed node the surface balances (convection + radiation under the wrap and pull blends, the
two-sided evaporation at the resolved chamber vapor pressure throttled by the salt, the spritz), the water flux
scaled by the exposed-area factor; and the surface states per exposed node: Maillard bark, smoke deposition, and
the smoke-ring front advancing along the node's inward line while the meat at the front is below myoglobin
denaturation. Returns the heat drawn from the pit [W] (the exposed-area factor included, as the slab's).
"""
function _piece_rhs!(pc::GridPiece, du, u, t, θ, Tp, ṁ, pv_star, airflow, t_wrap, t_pull, t_spritz, dt_spritz, m_spritz, hoisted, A_meat)
    mesh, n_s, n_a, v0, wbone, mats, mat, v0n = hoisted
    cfg = pc.cfg; N = mesh.N; nb = length(mesh.boundary)
    D_eff, k_m, h_c, L, k_mult, k_e, a_myosin, m_conv = θ
    g_fac = (airflow / cfg.g_ref)^m_conv
    h_c = h_c * g_fac; k_m = k_m * g_fac
    T = @view u[1:N]; W = @view u[(N + 1):2N]; C = @view u[(2N + 1):3N]; F = @view u[(3N + 1):4N]; Sl = @view u[(4N + 1):5N]
    dT = @view du[1:N]; dW = @view du[(N + 1):2N]; dC = @view du[(2N + 1):3N]; dF = @view du[(3N + 1):4N]; dSl = @view du[(4N + 1):5N]
    fill!(dT, zero(eltype(du))); fill!(dW, zero(eltype(du))); fill!(dSl, zero(eltype(du)))
    # the nodes' properties (cached once per call): conductivity, heat capacity of the node [J/K] and stretch
    Tel = eltype(du)
    kn = Vector{Tel}(undef, N); ρcv = Vector{Tel}(undef, N); st = Vector{Tel}(undef, N)
    vsum = zero(Tel)
    v0sum = 0.0
    @inbounds for n in 1:N
        mc = mats[mat[n]]
        v = specific_volume(W[n], F[n], mc)
        wb = wbone[n]
        kn[n] = k_mult * ((1 - wb) * conductivity_mix(W[n], F[n], T[n], v, mc) + wb * cfg.k_bone)
        st[n] = (v / v0n[n])^n_s
        ρcv[n] = ((1 - wb) * heat_capacity_dry(W[n], F[n], T[n], mc) / v + wb * cfg.ρc_bone) * mesh.vol[n] * st[n]
        vsum += mesh.vol[n] * v
        v0sum += mesh.vol[n] * v0n[n]
    end
    a_area = (vsum / v0sum)^n_a
    @inbounds for (i, j, g) in mesh.edges
        f = (st[i] + st[j]) / 2
        q = g * (kn[i] + kn[j]) / 2 * (T[j] - T[i]) / f                                 # W, i ← j
        dT[i] += q; dT[j] -= q
        jw = g * D_eff * (W[j] - W[i]) / f^2                                            # the slab's dry-mass form
        dW[i] += jw; dW[j] -= jw
        js = g * cfg.D_salt * (Sl[j] - Sl[i]) / f^2
        dSl[i] += js; dSl[j] -= js
    end
    @inbounds for n in 1:N
        dW[n] /= mesh.vol[n]
        dSl[n] /= mesh.vol[n]
        mc = mats[mat[n]]
        dC[n] = collagen_rate(T[n], mc) * (1 - C[n])
        dW[n] -= expulsion_rate_salted(W[n], T[n], Sl[n], k_e, a_myosin, mc)
        dF[n] = -fat_rate(T[n], mc) * F[n]
    end
    # the exposed nodes: heat in, water out, the surface states
    s = wrap_blend(t, t_wrap, cfg)
    pl = pull_blend(t, t_pull, cfg)
    j_sp = spritz_rate(t, t_spritz, dt_spritz, m_spritz, cfg) * (1 - s) * (1 - pl)
    scale = A_meat * cfg.n_meat * a_area
    sm = max(ṁ, zero(ṁ))
    fire = sm / (sm + cfg.mdot_ring)
    o_b = 5N; o_s = 5N + nb; o_r = 5N + 2nb
    q_meat = zero(Tel)
    @inbounds for (b, (n, area)) in enumerate(mesh.boundary)
        Ts, Ws = T[n], W[n]; mc = mats[mat[n]]
        f_aw = 1 / (1 + cfg.k_aw * max(Sl[n], zero(Sl[n])))
        j = surface_j_blended(Ts, Ws, h_c, k_m, s, pl, pv_star, zero(pv_star), mc, f_aw)
        q_pit = surface_q_pit_blended(Tp, Ts, h_c, s, pl, mc)
        q_in = q_pit + pl * surface_heat_rest(Ts, mc)
        q_net = q_in - cfg.λ_vap * (j + (1 - cfg.f_abs) * j_sp) - CP_WATER * (Ts - cfg.T_spritz) * j_sp
        dT[n] += area * q_net
        dW[n] -= area * a_area * (j - cfg.f_abs * j_sp) / (RHO_DRY * mesh.vol[n])
        q_meat += scale * area * q_pit
        aw = a_w_oswin(Ws, mc)
        wet = wet_blend(aw, mc)
        B = u[o_b + b]; Sm = u[o_s + b]; Rg = u[o_r + b]
        du[o_b + b] = bark_rate(Ts, aw, mc) * (1 - B)
        du[o_s + b] = cfg.k_dep * wet * sm * (1 - s) * (1 - pl) * (1 - Sm)
        T_front = _grid_T_at_depth(T, Rg, mesh.inward[b], mesh.inward_d[b])
        du[o_r + b] = cfg.D_ring * wet * fire * (1 - s) * (1 - pl) *
                      logistic((cfg.T_ring - T_front) / cfg.δ_ring) / (Rg + cfg.r_ring0)
    end
    @inbounds for n in 1:N
        dT[n] /= ρcv[n]
    end
    return q_meat
end

# the temperature at depth `d` along an inward line of nodes at depths `dl` (the slab's `temperature_at_depth`,
# clamped to the line; a line of one node, an unresolved direction, reads its node)
function _grid_T_at_depth(T, d, line, dl)
    n = length(line)
    n == 1 && return T[line[1]]
    dc = clamp(d, zero(d) + dl[1], zero(d) + dl[n])
    k = 1
    while k < n - 1 && _primal(dc) > dl[k + 1]
        k += 1
    end
    f = (dc - dl[k]) / (dl[k + 1] - dl[k])
    return (1 - f) * T[line[k]] + f * T[line[k + 1]]
end

"""
    piece_outputs(pc::GridPiece, Ap, θ, A_m, ps, tsave, ev)

The grid piece's trajectories: the core node's temperature, the area-weighted surface temperature and
evaporation, the fields (`T_field` etc. `N × nt`, `salt_field`), the through-thickness line at the core (`T_line`,
`W_line`, the reduced-order projection), the bark's, the smoke's and the ring's area-weighted means and the bark's
spread (the distributions per exposed node in `bark_dist`, `smoke_dist`, `ring_dist`), the exposed-area factor,
the physical through-thickness length along the core line, the heat load, the mass on a scale (per copy) and the
terminal quality integrals the slab reports.
"""
function piece_outputs(pc::GridPiece, Ap::AbstractMatrix, θ, A_m, ps::NamedTuple, tsave, ev::NamedTuple)
    mesh = pc.mesh; cfg = pc.cfg; N = mesh.N; nb = length(mesh.boundary); nt = length(tsave)
    T = Ap[1:N, :]; W = Ap[(N + 1):2N, :]; C = Ap[(2N + 1):3N, :]; F = Ap[(3N + 1):4N, :]; Sl = Ap[(4N + 1):5N, :]
    B = Ap[(5N + 1):(5N + nb), :]; S = Ap[(5N + nb + 1):(5N + 2nb), :]; Rg = Ap[(5N + 2nb + 1):(5N + 3nb), :]
    bnodes = first.(mesh.boundary); bareas = last.(mesh.boundary)
    wa = bareas ./ mesh.area
    Tsurf = vec(wa' * T[bnodes, :]); Wsurf = vec(wa' * W[bnodes, :])
    s_ts = wrap_blend.(tsave, ev.t_wrap, Ref(cfg)); pl_ts = pull_blend.(tsave, ev.t_pull, Ref(cfg))
    g_fac = (ps.airflow ./ cfg.g_ref) .^ θ[8]
    h_ts = θ[3] .* g_fac; km_ts = θ[2] .* g_fac
    n_s, n_a, v0 = shrink_exponents(cfg)
    mats, mat = pc.mats, pc.mat
    v0n = [specific_volume(mats[m].W0, mats[m].F0, mats[m]) for m in mat]
    a_ts = [_grid_area_factor(mesh, view(W, :, k), view(F, :, k), mats, mat, v0n, n_a) for k in 1:nt]
    j_evap = zeros(promote_type(eltype(Ap), eltype(ps.pv_star)), nt); j_pit = similar(j_evap); q_meat = similar(j_evap)
    for k in 1:nt
        jv = 0.0 * j_evap[k]; jp = jv; qm = jv
        for (b, n) in enumerate(bnodes)
            f_aw = 1 / (1 + cfg.k_aw * max(Sl[n, k], 0.0))
            fl = surface_fluxes_blended(T[n, k], W[n, k], h_ts[k], km_ts[k], s_ts[k], pl_ts[k], ps.pv_star[k], 0.0, mats[mat[n]], f_aw)
            jv += wa[b] * fl.j; jp += wa[b] * fl.j_pit
            qm += bareas[b] * surface_q_pit_blended(ps.Tpit[k], T[n, k], h_ts[k], s_ts[k], pl_ts[k], cfg)
        end
        j_evap[k] = jv; j_pit[k] = jp; q_meat[k] = A_m * cfg.n_meat * a_ts[k] * qm
    end
    bark_b = vec(wa' * B); bark_s = vec(wa' * S); ring = vec(wa' * Rg)
    bark_std = [sqrt(max(dot(wa, (B[:, k] .- bark_b[k]) .^ 2), 0.0)) for k in 1:nt]
    vw = mesh.vol ./ sum(mesh.vol)
    W0n = [mats[m].W0 for m in mat]; F0n = [mats[m].F0 for m in mat]
    W0bar = dot(vw, W0n); F0bar = dot(vw, F0n)                       # the cut's initial means (each node its material's)
    w_retained = dot(vw, W[:, end]) / W0bar
    F_mean = dot(vw, F[:, end])
    R_ret = (1 - cfg.η_drain) * (F0bar - F_mean)
    dts = diff(collect(float.(tsave)))
    cumtrap(v) = vcat(zero(eltype(v)), cumsum((v[1:(end - 1)] .+ v[2:end]) ./ 2 .* dts))
    evap_w_cum = cumtrap(j_evap .* a_ts .* mesh.area) ./ (RHO_DRY * sum(mesh.vol))
    jsp = spritz_rate.(tsave, ev.t_spritz, ev.dt_spritz, ev.m_spritz, Ref(cfg)) .* (1 .- s_ts) .* (1 .- pl_ts) .* a_ts
    spritz_w_cum = cfg.f_abs .* cumtrap(jsp .* mesh.area) ./ (RHO_DRY * sum(mesh.vol))
    dep_dirty = zeros(promote_type(eltype(Ap), eltype(ps.mdot)), nt)
    for k in 1:nt
        for (b, n) in enumerate(bnodes)
            dep_dirty[k] += wa[b] * cfg.k_dep * wet_blend(a_w_oswin(W[n, k], cfg), cfg) * max(ps.mdot[k], 0.0) *
                            (1 - ps.smoke_clean[k]) * (1 - s_ts[k]) * (1 - pl_ts[k]) * (1 - S[b, k])
        end
    end
    bitterness_cum = cumtrap(dep_dirty)
    evap_w, spritz_w, bitterness = evap_w_cum[end], spritz_w_cum[end], bitterness_cum[end]
    mass = [RHO_DRY * A_m * sum(mesh.vol[n] * ((1 - F0n[n]) + F[n, k] + (1 - cfg.η_drain) * (F0n[n] - F[n, k]) + W[n, k]) for n in 1:N) for k in 1:nt]
    # the physical length of the core line (core to the nearest exposed face): its raw spacings stretched by the
    # mean of each pair's node stretches
    line = mesh.line
    raw = [sqrt(sum(abs2, mesh.coords[line[i + 1], :] .- mesh.coords[line[i], :])) for i in 1:(length(line) - 1)]
    stn(n, k) = (specific_volume(W[n, k], F[n, k], mats[mat[n]]) / v0n[n])^n_s
    l_thick = [sum(raw[i] * (stn(line[i], k) + stn(line[i + 1], k)) / 2 for i in eachindex(raw); init=0.0 * a_ts[k]) for k in 1:nt]
    return (; id=pc.id, Tsurf, Tcore=T[mesh.core, :], bark_b, bark_s, bark_std, ring, l_star=l_star_bark.(bark_b, bark_s, Ref(cfg)),
            j_evap, j_pit, q_meat, area_factor=a_ts, vapor_load=A_m * cfg.n_meat * mesh.area .* a_ts .* j_pit,
            C_field_final=C[:, end], w_retained, juiciness=w_retained + cfg.β_fat * R_ret / W0bar, fat_rendered=(F0bar - F_mean) / F0bar,
            evap_w, spritz_w, drip_w=(W0bar - w_retained * W0bar) + spritz_w - evap_w, drip_f=cfg.η_drain * (F0bar - F_mean),
            evap_w_cum, spritz_w_cum, bitterness_cum, l_thick,
            T_field=T, W_field=W, C_field=C, F_field=F, salt_field=Sl, wrap_effective=s_ts,
            bitterness, mass, T_line=T[mesh.line, :], W_line=W[mesh.line, :], C_line=C[mesh.line, :], Tcold=vec(minimum(T; dims=1)),
            bark_dist=B, smoke_dist=S, ring_dist=Rg)
end
