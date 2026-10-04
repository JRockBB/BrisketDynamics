# The voxel piece (src/meats/voxel3d.jl), gated:
#   geometry     a voxelized sphere's exposed area within 3 % of 4πR² (the smooth surface, not the staircase's) and its
#                volume within 1 %; a box's exposed area within 4 % of its six faces (its sharp edges rounded by a cell)
#   convergence  a box under a prescribed pit: the core's and the mean temperature against Brick3D's close as the
#                cells shrink
#   conservation the water the exposed faces lose and the sinks take is what the nodes lose, from the RHS itself
#   materials    fat-cap cells take the cap's composition; bone cells are thermal-only bone

using SparseArrays: findnz

const V_HR = 3600.0
const V_θ = [3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6]

# labels of a box of a × b × c cells resting on the board (the first layer), centred in a grid with a margin
function box_labels(a, b, c; margin=3)
    L = zeros(Int8, a + 2margin, b + 2margin, c + margin)
    L[(margin + 1):(margin + a), (margin + 1):(margin + b), 1:c] .= 1
    return L
end

@testset "voxel: geometry of a sphere and a box" begin
    h = 0.002; R = 0.06; n = ceil(Int, R / h) + 3
    L = zeros(Int8, 2n, 2n, 2n + 4)                                   # floating clear of the board
    c = (n * h, n * h, (n + 4) * h)
    for I in CartesianIndices(L)
        x = ((I[1] - 0.5) * h, (I[2] - 0.5) * h, (I[3] - 0.5) * h)
        sum(abs2, x .- c) <= R^2 && (L[I] = 1)
    end
    pc = VoxelPiece(:s, L, h; coarsen=2)
    @test abs(pc.mesh.area / (4π * R^2) - 1) < 0.03
    @test abs(sum(pc.mesh.vol) / (4π / 3 * R^3) - 1) < 0.01
    @test pc.mesh.depth[pc.mesh.core] > 0.85R && pc.mesh.line[1] == pc.mesh.core   # the core near the centre
    @test pc.kind === :voxel3d && all(==(1), pc.mat) && all(==(0.0), pc.bone)
    # a box: its six faces (the board's under it), the edges rounded by about a cell
    Lb = box_labels(30, 20, 15)
    pb = VoxelPiece(:b, Lb, h; coarsen=2)
    a, b, cc = 30h, 20h, 15h
    @test abs(pb.mesh.area / (2(a * b + b * cc + cc * a)) - 1) < 0.04
    @test abs(sum(pb.mesh.vol) / (a * b * cc) - 1) < 1e-12
    @test length(pb.mesh.inward) == length(pb.mesh.boundary) && all(d -> d[1] > 0 && all(>(0), diff(d)), pb.mesh.inward_d)
end

@testset "voxel: a box converges to the brick under a prescribed pit" begin
    cfg = MeatConfig(; n_shrink=0.0, T0=282.0, p_v_pit=2000.0)
    tknots = collect(0.0:600.0:(1.5V_HR)); Tpit = fill(383.15, length(tknots)); tsave = tknots
    hf = 0.001; a, b, c = 48, 32, 24                                  # a 48 × 32 × 24 mm box
    brick = Brick3D(:r; Lx=a * hf, Ly=b * hf, Lz=c * hf, nx=13, ny=9, nz=7, exposed=(true, true, true, true, true, true), cfg=cfg)
    meanT(o, m) = sum(m.vol .* o.T_field[:, end]) / sum(m.vol)
    rb = piece_solve(brick, V_θ, Tpit, tknots, tsave; pv=cfg.p_v_pit)
    errs = map((8, 4)) do k                                           # 8 mm cells, then 4 mm
        pc = VoxelPiece(:v, box_labels(a, b, c; margin=k), hf; coarsen=k, cfg=cfg)
        o = piece_solve(pc, V_θ, Tpit, tknots, tsave; pv=cfg.p_v_pit)
        (abs(o.Tcore[end] - rb.Tcore[end]), abs(meanT(o, pc.mesh) - meanT(rb, brick.mesh)))
    end
    println("  voxel box vs brick at 1.5 h, core / mean: 8 mm cells ", round.(errs[1]; digits=3), " K, 4 mm ", round.(errs[2]; digits=3), " K")
    @test errs[2][1] < errs[1][1] && errs[2][2] < errs[1][2] && errs[2][1] < 1.5 && errs[2][2] < 1.5
end

@testset "voxel: water is conserved; fat and bone take their materials" begin
    cfg = MeatConfig(; n_shrink=0.0, T0=282.0)
    h = 0.004
    L = box_labels(8, 6, 5; margin=1)
    L[2:9, 2:7, 5] .= 2                                               # a fat cap on top
    L[4:6, 3:5, 2] .= 3                                               # a bone inside
    pc = VoxelPiece(:w, L, h; coarsen=1, cfg=cfg)
    fat = findall(==(2), pc.mat); bone = findall(==(1.0), pc.bone)
    @test length(fat) == 48 && length(bone) == 9
    hoisted = BrisketDynamics._piece_hoisted(pc)
    @test all(hoisted[5][bone] .== 1.0) && all(hoisted[5][setdiff(1:pc.mesh.N, bone)] .== 0.0)
    u0 = initial_state(pc, V_θ)
    N = pc.mesh.N; nb = n_boundary(pc)
    @test all(u0[N .+ fat] .== BrisketDynamics.FAT_CAP_W0) && all(u0[3N .+ fat] .== BrisketDynamics.FAT_CAP_F0)
    # the RHS: the dry-mass water change summed over the nodes is the faces' loss plus the sinks
    x = fill(330.0, N); w = u0[(N + 1):2N]; cc = zeros(N); ff = u0[(3N + 1):4N]
    ucur = vcat(x, w, cc, ff, zeros(N), zeros(nb), zeros(nb), zeros(nb))
    du = zeros(length(ucur))
    θt = ntuple(j -> V_θ[j], 8); pv = 2000.0
    BrisketDynamics._piece_rhs!(pc, du, ucur, 600.0, θt, 380.0, 0.0, pv, cfg.g_ref, BrisketDynamics.WRAP_NEVER, BrisketDynamics.PULL_NEVER,
                                BrisketDynamics.SPRITZ_NEVER, 3600.0, 0.05, hoisted, 1.0)
    dW_total = sum(pc.mesh.vol .* du[(N + 1):2N])
    loss = sum(area * BrisketDynamics.surface_j_blended(x[n], w[n], V_θ[3], V_θ[2], 0.0, 0.0, pv, 0.0, pc.mats[pc.mat[n]]) / BrisketDynamics.RHO_DRY
               for (n, area) in pc.mesh.boundary)
    sinks = sum(pc.mesh.vol[n] * BrisketDynamics.expulsion_rate(w[n], x[n], V_θ[6], V_θ[7], pc.mats[pc.mat[n]]) for n in 1:N)
    @test abs(dW_total + loss + sinks) < 1e-10 * max(abs(loss), abs(sinks), 1e-9)
end

@testset "sparse Jacobian: the flat solved sparse agrees with dense" begin
    θ = [3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6]; φ = [15e3, 20.0, 3e-4, 1.2e7, 0.2, 3e-4, 0.85]
    pit = PitConfig(; T_amb=300.0, Tpit0=320.0)
    ts = collect(0.0:900.0:(4V_HR)); u = fill(0.35, 4); utk = collect(range(0.0, 4V_HR; length=4))
    sys = CookSystem(BrisketFlat3D(:f); pit)
    run(j) = system_solve(sys, [θ], [2.0], φ, u, utk, ts; abstol=1e-9, reltol=1e-9, jacobian=j)
    od, os = run(:dense), run(:sparse)
    @test maximum(abs.(od.piece[:f].T_field .- os.piece[:f].T_field)) < 1e-5
    @test maximum(abs.(od.piece[:f].W_field .- os.piece[:f].W_field)) < 1e-7 && maximum(abs.(od.Tpit .- os.Tpit)) < 1e-6
    # the pattern covers the frozen right-hand side: its coloured Jacobian equals a dense one of the frozen RHS
    nu = length(u); p, play = BrisketDynamics.pack_params(sys, [θ], [2.0], φ, u)
    x = BrisketDynamics.initial_state(sys, p, play) .+ 0.0
    proto, jac!, ncol = BrisketDynamics.make_sparse_jacobian(sys, 0.0, utk[2], nu)
    jac!(proto, x, p, 600.0)
    rf!, _ = BrisketDynamics.make_system_rhs(sys, 0.0, utk[2], nu; freeze=true)
    Jd = ForwardDiff.jacobian((dx, xx) -> rf!(dx, xx, p, 600.0), similar(x), x)
    Jd_on = [Jd[i, j] for (i, j) in zip(findnz(proto)[1:2]...)]
    @test maximum(abs.(Jd_on .- findnz(proto)[3])) <= 1e-9 * maximum(abs.(Jd)) && ncol < 100
    # nothing of the frozen RHS falls outside the pattern but the pit's draw on the meat (its rows, left out by design)
    off = abs.(Jd .- Matrix(proto)); pitrows = collect(sys.layout.pit_range)
    off[pitrows, :] .= 0
    @test maximum(off) <= 1e-9 * maximum(abs.(Jd))
end
