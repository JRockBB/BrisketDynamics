# The grid pieces (src/meats/grid3d.jl), gated:
#   the 1-D limit    a brick of one transverse node with one exposed face reproduces the slab (n_shrink = 0) in a system
#   symmetry         a cube with every face exposed gives the same field under any axis permutation
#   conservation     the water the exposed faces lose and the sinks take is what the nodes lose, from the RHS itself
#   the cylinder     the axisymmetric piece under a prescribed pit against the ADI engine (meat_solve_2d)
#   the infinite one the cylinder with insulated ends against the slab's cylindrical mode (geom_m = 1)
#   restart          a grid piece hands off and restarts in a system
using LinearAlgebra

const G_HR = 3600.0
const G_θ = [3.8e-9, 0.0018, 10.0, 0.03, 1.0, 5.6e-4, 0.20, 0.6]
const G_φ = [15e3, 20.0, 3e-4, 1.2e7, 0.2, 3e-4, 0.85]
const G_PIT = PitConfig(; T_amb=300.0, Tpit0=320.0)

@testset "grid: meshes and layouts" begin
    b = Brick3D(:b; Lx=0.03, Ly=0.1, Lz=0.2, nx=4, ny=3, nz=5, exposed=(false, true, true, true, true, true))
    m = b.mesh
    @test m.N == 60 && m.dims == (4, 3, 5) && m.extents == (0.03, 0.1, 0.2) && n_nodes(b) == 60
    @test abs(sum(m.vol) - 0.03 * 0.1 * 0.2) < 1e-15                              # the half cells tile the box
    @test abs(m.area - (0.1 * 0.2 + 2 * 0.03 * 0.2 + 2 * 0.03 * 0.1)) < 1e-15    # five faces exposed
    @test length(m.edges) == 3 * 3 * 5 + 4 * 2 * 5 + 4 * 3 * 4
    @test m.coords[m.core, 1] == 0.0 && b.cfg.A_meat == 1.0
    @test ode_layout(b).n == 4 * 60 + 2 * n_boundary(b) && ode_layout(b)[:bark_b] == (241):(240 + n_boundary(b))
    @test_throws ArgumentError initial_state(b, G_θ; ics=(T0f=fill(NaN, b.mesh.N),))
    @test_throws ArgumentError Brick3D(:b; Lx=0.03, Ly=0.1, Lz=0.2, nx=4, ny=3, nz=5, exposed=(false, false, false, false, false, false))
    @test_throws ArgumentError Brick3D(:b; Lx=-0.03, Ly=0.1, Lz=0.2, nx=4, ny=3, nz=5)
    @test_throws ArgumentError Brick3D(:b; Lx=0.03, Ly=0.1, Lz=0.2, nx=1, ny=1, nz=1)
    c = Cylinder2D(:c; R=0.05, Hz=0.09, nr=6, nz=7)
    @test c.mesh.N == 42 && abs(sum(c.mesh.vol) - π * 0.05^2 * 0.09) < 1e-15 && abs(c.mesh.area - (2π * 0.05 * 0.09 + 2π * 0.05^2)) < 1e-15
    ci = Cylinder2D(:c; R=0.05, Hz=0.09, nr=6, nz=7, exposed_ends=false)
    @test abs(ci.mesh.area - 2π * 0.05 * 0.09) < 1e-15 && length(ci.mesh.line) == 6
    @test_throws ArgumentError Cylinder2D(:c; R=0.05, Hz=0.09, nr=1, nz=7)
    @test_throws ArgumentError Cylinder2D(:c; R=NaN, Hz=0.09, nr=6, nz=7)
end

@testset "grid: three-dimensional manufactured diffusion field" begin
    # For T = T₀ + a(x² + y² + z²), the continuum operator at an interior point is
    # div(k(T)∇T)/(ρc) = [6ak + 4a²(x²+y²+z²) dk/dT]/(ρc). The finite-volume
    # operator must converge to it as all three mesh directions are refined.
    cfg = MeatConfig(; n_shrink=0.0, T0=300.0)
    a = 750.0
    function manufactured_error(n)
        pc = Brick3D(:m; Lx=0.04, Ly=0.04, Lz=0.04, nx=n, ny=n, nz=n,
                     exposed=(true, true, true, true, true, true), cfg)
        mesh = pc.mesh; N = mesh.N; nb = n_boundary(pc)
        T = [cfg.T0 + a * sum(abs2, view(mesh.coords, i, :)) for i in 1:N]
        W = fill(cfg.W0, N); C = zeros(N); F = fill(cfg.F0, N)
        u = vcat(T, W, C, F, zeros(nb), zeros(nb)); du = zeros(length(u))
        θ = ntuple(i -> G_θ[i], 8)
        BrisketDynamics._piece_rhs!(pc, du, u, 0.0, θ, cfg.T0, 0.0, cfg.p_v_pit, cfg.g_ref,
                                    BrisketDynamics.WRAP_NEVER, BrisketDynamics.PULL_NEVER,
                                    BrisketDynamics.SPRITZ_NEVER, 3600.0, 0.0, Inf,
                                    BrisketDynamics._piece_hoisted(pc), 1.0)
        centre = argmin([sum(abs2, view(mesh.coords, i, :) .- 0.02) for i in 1:N])
        x = view(mesh.coords, centre, :); Tc = T[centre]
        v = BrisketDynamics.specific_volume(cfg.W0, cfg.F0, cfg)
        km(t) = G_θ[5] * BrisketDynamics.conductivity_mix(cfg.W0, cfg.F0, t, v, cfg)
        ρc = BrisketDynamics.heat_capacity_dry(cfg.W0, cfg.F0, Tc, cfg) / v
        exact = (6a * km(Tc) + 4a^2 * sum(abs2, x) * ForwardDiff.derivative(km, Tc)) / ρc
        return abs(du[centre] - exact)
    end
    coarse = manufactured_error(5)
    fine = manufactured_error(9)
    @test fine < coarse / 3
    @test fine < 2e-5
end

@testset "grid: the one-dimensional limit is the slab" begin
    L = G_θ[4]; nx = 12
    cfg = MeatConfig(; nx=nx, n_shrink=0.0, A_meat=1.0, T0=282.0)
    ts = collect(0.0:600.0:(4G_HR)); u = [0.2, 0.5, 0.4, 0.3, 0.3]; utk = collect(range(0.0, 4G_HR; length=5))
    tol = 1e-9
    ev = [CookEvent(:wrap, :s, 2.5G_HR), CookEvent(:spritz, :s, 1G_HR; dt=1800.0, m=0.04)]
    slab = system_solve(CookSystem(Slab1D(:s; cfg); pit=G_PIT), [G_θ], [1.0], G_φ, u, utk, ts; events=ev, abstol=tol, reltol=tol)
    brick = Brick3D(:s; Lx=L, Ly=1.0, Lz=1.0, nx=nx, ny=1, nz=1, exposed=(false, true, false, false, false, false), cfg=cfg)
    @test brick.mesh.area == 1.0 && brick.mesh.line == collect(1:nx) && brick.mesh.core == 1
    grid = system_solve(CookSystem(brick; pit=G_PIT), [G_θ], [1.0], G_φ, u, utk, ts; events=ev, abstol=tol, reltol=tol)
    ps, pg = slab.piece[:s], grid.piece[:s]
    @test maximum(abs.(pg.T_field .- ps.T_field)) < 1e-4
    @test maximum(abs.(pg.W_field .- ps.W_field)) < 1e-6
    @test maximum(abs.(pg.C_field .- ps.C_field)) < 1e-6 && maximum(abs.(pg.F_field .- ps.F_field)) < 1e-7
    @test maximum(abs.(pg.bark_b .- ps.bark_b)) < 1e-5 && maximum(abs.(pg.bark_s .- ps.bark_s)) < 1e-5
    @test maximum(abs.(grid.Tpit .- slab.Tpit)) < 1e-3 && maximum(abs.(grid.pv_pit .- slab.pv_pit)) < 1e-2
    @test maximum(abs.(pg.q_meat .- ps.q_meat)) < 1e-2 && maximum(abs.(pg.j_evap .- ps.j_evap)) < 1e-8
    @test maximum(abs.(pg.mass .- ps.mass)) < 1e-6 && abs(pg.w_retained - ps.w_retained) < 1e-6
    @test maximum(abs.(pg.T_line .- ps.T_field)) < 1e-4 && maximum(abs.(pg.Tcore .- ps.Tcore)) < 1e-4
end

@testset "grid: a cube's field is invariant under axis permutation; water is conserved" begin
    cfg = MeatConfig(; n_shrink=0.0, T0=282.0)
    ts = collect(0.0:600.0:(1.5G_HR)); u = fill(0.3, 3); utk = collect(range(0.0, 1.5G_HR; length=3))
    cube = Brick3D(:c; Lx=0.04, Ly=0.04, Lz=0.04, nx=5, ny=5, nz=5, exposed=(true, true, true, true, true, true), cfg=cfg)
    o = system_solve(CookSystem(cube; pit=G_PIT), [G_θ], [1.0], G_φ, u, utk, ts)
    T = reshape(o.piece[:c].T_field[:, end], 5, 5, 5)
    @test maximum(abs.(T .- permutedims(T, (2, 1, 3)))) < 1e-9 && maximum(abs.(T .- permutedims(T, (3, 2, 1)))) < 1e-9
    @test maximum(abs.(T .- T[end:-1:1, :, :])) < 1e-9                              # and under reflection
    @test T[3, 3, 3] < T[1, 3, 3] && T[1, 1, 1] > T[1, 3, 3]                        # the core coolest, the corner hottest
    @test cube.mesh.core == 63 && o.piece[:c].Tcore[end] == T[3, 3, 3]
    # conservation from the RHS: the dry-mass water change summed over the nodes equals the faces' loss plus the sinks
    hoisted = BrisketDynamics._piece_hoisted(cube)
    x = o.piece[:c].T_field[:, 4]; w = o.piece[:c].W_field[:, 4]; cc = o.piece[:c].C_field[:, 4]; ff = o.piece[:c].F_field[:, 4]
    nb = n_boundary(cube)
    ucur = vcat(x, w, cc, ff, zeros(nb), zeros(nb))
    du = zeros(length(ucur))
    θt = ntuple(j -> G_θ[j], 8)
    pv = 2000.0
    BrisketDynamics._piece_rhs!(cube, du, ucur, ts[4], θt, 380.0, 0.0, pv, cfg.g_ref, BrisketDynamics.WRAP_NEVER, BrisketDynamics.PULL_NEVER,
                                BrisketDynamics.SPRITZ_NEVER, 3600.0, 0.05, Inf, hoisted, 1.0)
    N = cube.mesh.N
    dW_total = sum(cube.mesh.vol .* du[(N + 1):2N])                                  # m³ × d(kg/kg)/dt
    loss = 0.0
    for (n, area) in cube.mesh.boundary
        loss += area * BrisketDynamics.surface_j_blended(x[n], w[n], G_θ[3], G_θ[2], 0.0, 0.0, pv, 0.0, cfg) / BrisketDynamics.RHO_DRY
    end
    sinks = sum(cube.mesh.vol[n] * BrisketDynamics.expulsion_rate(w[n], x[n], G_θ[6], G_θ[7], cfg) for n in 1:N)
    @test abs(dW_total + loss + sinks) < 1e-12 * max(abs(loss), abs(sinks), 1e-9)
    @test sum(cube.mesh.vol .* du[(2N + 1):3N]) > 0                                   # collagen only converts
end

@testset "grid: the axisymmetric piece against the ADI engine and the slab's cylindrical mode" begin
    cfg = MeatConfig(; n_shrink=0.0, T0=282.0, p_v_pit=2000.0)
    R, Hz, nr, nz = 0.04, 0.10, 10, 12
    θ2 = G_θ[[1, 2, 3, 5, 6, 7]]
    tknots = collect(0.0:600.0:(2G_HR)); Tpit = fill(383.15, length(tknots)); tsave = tknots
    adi = meat_solve_2d(θ2, Tpit, tknots, tsave; R, Hz, nr, nz, cfg, h=10.0)
    cyl = Cylinder2D(:c; R, Hz, nr, nz, cfg)
    θc = copy(G_θ); θc[8] = 0.0                                                     # the ADI takes a fixed h_c: no draft exponent
    mol = piece_solve(cyl, θc, Tpit, tknots, tsave; pv=cfg.p_v_pit)
    dcore = maximum(abs.(mol.Tcore .- adi.Tcore))
    Tm = reshape(mol.T_field[:, end], nr, nz); Ta = adi.T_field[:, :, end]
    dfield = maximum(abs.(Tm .- Ta))
    Wm = reshape(mol.W_field[:, end], nr, nz); Wa = adi.W_field[:, :, end]
    adi20 = meat_solve_2d(θ2, Tpit, tknots, tsave; R, Hz, nr, nz, cfg, h=20.0)
    d20 = maximum(abs.(mol.Tcore .- adi20.Tcore)); w20 = maximum(abs.(reshape(mol.W_field[:, end], nr, nz) .- adi20.W_field[:, :, end]))
    println("  cylinder MOL vs ADI: core ", round(dcore; digits=3), " K (h = 10 s) / ", round(d20; digits=3), " K (h = 20 s), field ", round(dfield; digits=3),
            " K, water ", round(maximum(abs.(Wm .- Wa)); digits=4), " / ", round(w20; digits=4))
    # the gap is the ADI's own and does not move with its step (0.33 K at 10 and 20 s): its rim balance takes a slab-like
    # half-cell capacity on a cylindrical rim, a 3 % effect at ten radial nodes, and its face evaporation is linearized-implicit;
    # the cylinder's discretization is gated below against the slab's cylindrical mode, to 1e-3 K
    @test dcore < 0.5 && dfield < 1.0 && maximum(abs.(Wm .- Wa)) < 0.06
    @test abs(dcore - d20) < 0.1 && abs(maximum(abs.(Wm .- Wa)) - w20) < 0.01
    @test abs(mol.w_retained - adi.w_retained) < 0.01
    # the ends insulated: the infinite cylinder, the slab with geom_m = 1 on the same radial nodes
    cyl_inf = Cylinder2D(:c; R, Hz, nr, nz, exposed_ends=false, cfg)
    mol_inf = piece_solve(cyl_inf, θc, Tpit, tknots, tsave; pv=cfg.p_v_pit, abstol=1e-10, reltol=1e-10)
    cfg1 = with_config(cfg; nx=nr, geom_m=1.0, A_meat=1.0)
    θ1 = copy(θc); θ1[4] = R
    sl = meat_solve(θ1, Tpit, tknots, tsave; cfg=cfg1, abstol=1e-10, reltol=1e-10)
    @test maximum(abs.(mol_inf.T_line .- sl.T_field)) < 1e-3
    @test maximum(abs.(mol_inf.W_line .- sl.W_field)) < 1e-5
    @test maximum(abs.(mol_inf.Tcore .- sl.Tcore)) < 1e-3
    # every row of the insulated cylinder is the same (no axial gradient)
    Ti = reshape(mol_inf.T_field[:, end], nr, nz)
    @test maximum(abs.(Ti .- Ti[:, 1])) < 1e-9
end

@testset "grid: a piece restarts in a system, beside a slab" begin
    cfg = MeatConfig(; n_shrink=0.0, T0=282.0)
    ts = collect(0.0:600.0:(2G_HR)); u = fill(0.3, 3); utk = collect(range(0.0, 2G_HR; length=3))
    brick = Brick3D(:b; Lx=0.03, Ly=0.08, Lz=0.12, nx=6, ny=4, nz=5, cfg=cfg)
    slab = Slab1D(:s; cfg=MeatConfig(; nx=8))
    sys = CookSystem([slab, brick]; pit=G_PIT)
    @test sys.layout.pieces == [:s, :b] && n_state(sys) == (5 * 8 + 4) + (4 * brick.mesh.N + 2 * n_boundary(brick)) + 6
    whole = system_solve(sys, [G_θ, G_θ], [0.12, 1.0], G_φ, u, utk, ts)
    k = 7
    h = system_handoff(whole, k)
    @test haskey(h.pieces[2], :B0f) && length(h.pieces[2].B0f) == n_boundary(brick) && !haskey(h.pieces[2], :ring)
    second = system_solve(sys, [G_θ, G_θ], [0.12, 1.0], G_φ, u, utk, ts[k:end];
                          ics=h.pieces, pit_ics=(; h.pit..., ignited=h.ignited), history=h.history)
    @test maximum(abs.(second.piece[:b].Tcore .- whole.piece[:b].Tcore[k:end])) < 0.05
    @test maximum(abs.(second.piece[:b].bark_b .- whole.piece[:b].bark_b[k:end])) < 1e-6
    @test maximum(abs.(second.Tpit .- whole.Tpit[k:end])) < 0.05
    @test abs(second.piece[:b].evap_w - whole.piece[:b].evap_w) < 1e-4
    @test abs(second.piece[:b].bitterness - whole.piece[:b].bitterness) < 1e-4
    @test whole.q_meat ≈ whole.piece[:s].q_meat .+ whole.piece[:b].q_meat
    @test all(whole.piece[:b].bark_std .>= 0) && whole.piece[:b].bark_std[end] > 0                 # the bark is a distribution over the faces
end
