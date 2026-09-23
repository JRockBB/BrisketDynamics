# The filter on the graph (sequential.jl) and the block it lives on: the prior families and the block declared
# once, the state block's packing, the slice physics against the whole-cook physics at the recipe's state, the
# channel wrapper that puts a channel's discrepancy in the block, updates on arrival with the consistency
# diagnostics, the plan from the filtered state against the whole-cook solve (the restart contract), and the
# innovation misfit.
#   julia --project=estimator estimator/test/test_sequential.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
sites, names, prior_phys = blk.sites, blk.names, blk.prior_phys
AREA, DEPTH = blk.area, blk.depth
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
z0 = z_center.(sites)
x0 = from_z(z0[1:N_PHYS], prior_phys)
nc = RealCookNoise()
np = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
channels() = realcook_channels(readings, masks, np; cap=nc.hum_cap, depth_latent=DEPTH)
noise = [np[i] for i in 1:4]

@testset "the families: transforms, priors, the block declared once" begin
    lb = PriorSite("logbeta", 3.0, 5.0, -9.0, -8.0)
    ps = PriorSite("beta", 4.0, 2.0; coord=:score); lbs = PriorSite("logbeta", 3.0, 5.0, -9.0, -8.0; coord=:score)
    for p in (PriorSite("lognormal", -1.0, 0.3), PriorSite("beta", 4.0, 2.0), lb, PriorSite("linear", 300.0, 2.0), ps, lbs)
        for v in (-1.3, 0.0, 0.7)
            x = from_z(v, p)
            @test abs(to_z(x, p) - v) < 1e-9
            @test abs(dx_dz(v, p) - ForwardDiff.derivative(w -> from_z(w, p), v)) < 1e-9 * max(1, abs(dx_dz(v, p)))
        end
    end
    @test exp(-9.0) < from_z(-3.0, lb) < from_z(3.0, lb) < exp(-8.0)                  # bounded in the log range
    @test abs(from_z(z_center(lb), lb) - exp(-9.0 + 3 / 8)) < 1e-12                    # the centre is the Beta's mean fraction
    # the score: its centre is the Beta's median, it is bounded, it inverts, it is held beyond eight sd, its prior is N(0, 1)
    @test isscore(ps) && !isscore(lb) && z_center(ps) == 0.0 && z_moments(lbs) == (0.0, 1.0)
    @test abs(from_z(0.0, ps) - 0.6862) < 1e-3 && exp(-9.0) < from_z(-3.0, lbs) < from_z(3.0, lbs) < exp(-8.0)
    @test abs(E._score_of_fraction(E._score_fraction(1.7, 4.0, 2.0), 4.0, 2.0) - 1.7) < 1e-9
    @test from_z(9.0, ps) == from_z(8.0, ps) && dx_dz(9.0, ps) == 0.0
    @test prior_terms([1.3], [ps]) == (0.5 * 1.3^2, [1.3], [1.0])
    @test_throws ArgumentError PriorSite("lognormal", 0.0, 1.0; coord=:score)
    @test_throws ArgumentError PriorSite("beta", 1.0, 1.0; coord=:probit)
    pu = sites[names .== "u_leak"][1]                                                    # the pilot's prior carries its bounded sites in the score
    @test isscore(pu) && abs(drift_scale(pu) * sqrt(z_moments(PriorSite("beta", pu.p1, pu.p2))[2]) - 1) < 0.1 && drift_scale(sites[1]) == 1.0
    for p in (PriorSite("beta", 4.0, 2.0), lb, PriorSite("lognormal", -1.0, 0.3), ps, lbs)
        for v in (-0.4, 1.1)
            f, g, h = prior_terms([v], [p])
            fg(w) = prior_terms(w, [p])[1]
            @test abs(g[1] - ForwardDiff.gradient(fg, [v])[1]) < 1e-8
            @test abs(h[1] - ForwardDiff.hessian(fg, [v])[1, 1]) < 1e-8
        end
    end
    @test_throws ArgumentError PriorSite("gaussian", 0.0, 1.0)
    @test length(sites) == 17 && names[16] == "A_meat" && names[17] == "core_depth_fraction" && AREA == 16 && DEPTH == 17
    @test blk.idx["u_leak"] == 13 && blk.idx["a0"] == 15 && sites[13].family == "beta" &&
          sites[DEPTH].family == "beta" && isscore(sites[DEPTH]) && sites[1].family == "lognormal"
    @test read_sites([Dict("family" => "logbeta", "p1" => 2, "p2" => 3, "p3" => -1, "p4" => 1)])[1].p4 == 1.0
    @test read_sites([Dict("family" => "beta", "p1" => 2, "p2" => 3)])[1].p3 == 0.0
    serialized = Dict("family" => "logbeta", "p1" => 2.0, "p2" => 3.0, "p3" => -9.2, "p4" => -7.8, "coord" => "score")
    loaded = only(read_sites([serialized]))
    @test (loaded.family, loaded.p1, loaded.p2, loaded.p3, loaded.p4, loaded.coord) ==
          ("logbeta", 2.0, 3.0, -9.2, -7.8, :score)
    @test from_z(0.0, loaded) == from_z(0.0, PriorSite("logbeta", 2.0, 3.0, -9.2, -7.8; coord=:score))
    @test_throws ArgumentError read_sites([Dict("family" => "logbeta", "p1" => 2, "p2" => 3)])
    zt = draw_truth(MersenneTwister(3), sites)
    xt = from_z(zt, sites)
    @test length(zt) == 17 && all(isfinite, zt) && 0 < xt[DEPTH] < 1
    @test probe_depth(xt[4], xt[DEPTH]) < xt[4]
    rng_depth = MersenneTwister(31)
    depth_draws = [from_z(only(draw_truth(rng_depth, [sites[DEPTH]])), sites[DEPTH]) for _ in 1:200]
    @test all(x -> 0 < x < 1, depth_draws) && abs(sum(depth_draws) / length(depth_draws) - 0.8) < 0.03
    @test maximum(depth_draws) > 0.9                              # no artificial pile-up or clipping at 0.9 L
    @test_throws ArgumentError draw_truth(MersenneTwister(3), sites; depth=DEPTH)
end

@testset "the state block: packing, the linear family, the discrepancies" begin
    o0 = monolith_solve(x0[1:N_THETA], x0[(N_THETA + 1):N_PHYS], u[1:2], t[1:2], t[1:2]; pit, cfg, A_meat=exp(z0[AREA]), events...)
    ics, a = handoff(o0, 1)
    lay = FilterLayout(length(ics.T0f), length(sites), 4)
    @test lay.n_state == 5 * lay.nx + 9 && lay.n == 17 + lay.n_state + 4 && disc_index(lay, 1) == 17 + lay.n_state + 1 && last(disc_range(lay)) == lay.n
    xs = pack_state(ics, a, lay)
    ics2, a2 = unpack_state(xs, lay)
    @test isapprox(a2, a; atol=2e-8)
    @test isapprox(ics2.T0f, ics.T0f; atol=2e-8)
    @test maximum(abs.(ics2.Ssalt0f .- ics.Ssalt0f)) <= 2e-8
    @test isapprox(ics2.Tpit0, ics.Tpit0; atol=2e-8)
    @test isapprox(ics2.Tfb0, ics.Tfb0; atol=2e-8)
    @test isapprox(ics2.mfuel0, ics.mfuel0; atol=2e-8)
    xp = state_from_coord(xs, lay)
    @test xp[pit_index(lay) - lay.n_z] ≈ ics.Tpit0 && xp[centre_index(lay) - lay.n_z] ≈ ics.T0f[1] && xp[surface_index(lay) - lay.n_z] ≈ ics.T0f[end]
    kinds = E.state_kinds(lay)
    @test all(xp[i] > 0 for i in eachindex(xp) if kinds[i] == :positive)
    @test all(0 < xp[i] < 1 for i in eachindex(xp) if kinds[i] == :unit)
    for δ in (-5.0, 5.0)
        xd = state_from_coord(xs .+ δ, lay)
        @test all(xd[i] > 0 for i in eachindex(xd) if kinds[i] == :positive)
        @test all(0 < xd[i] < 1 for i in eachindex(xd) if kinds[i] == :unit)
    end
    ss = state_sites(lay.n_state)
    v = randn(MersenneTwister(1), lay.n_state)
    @test from_z(v, ss) == v && all(E.dx_dz(v, ss) .== 1)
    joint = vcat(sites, ss, disc_sites(4))
    f, g, h = prior_terms(vcat(z0, v, zeros(4)), joint)
    @test length(g) == lay.n && all(isfinite, h)
    sn = StateNoise()
    @test length(E.state_rates(sn, lay)) == lay.n_state && length(E.state_init_sd(sn, lay)) == lay.n_state
    @test E.state_rates(sn, lay)[1] == sn.T && E.state_rates(sn, lay)[end] == sn.ignited && E.state_init_sd(sn, lay)[5 * lay.nx + 4] == sn.T_pit0
    # the groups tile the state block once, in StateNoise's order, each at its own rate
    gr = state_groups(lay)
    @test first.(gr) == collect(STATE_GROUPS) && sort(vcat([collect(r) for (_, r) in gr]...)) == 1:lay.n_state
    @test all(E.state_rates(sn, lay)[first(r)] == getfield(sn, Symbol(g)) && E.state_init_sd(sn, lay)[last(r)] == getfield(sn, Symbol(g * "0")) for (g, r) in gr)
    @test pit_index(lay) - lay.n_z == first(gr[findfirst(x -> first(x) == "T_pit", gr)][2])
    # the noise's structure: under :offset a field's nodes share one draw (the block is rank one plus the jitter),
    # the scalars are independent; under :nodes everything is diagonal; both are positive definite
    Qo = state_noise_cov(lay, sn, E.state_init_sd(sn, lay)); Qn = state_noise_cov(lay, StateNoise(fields=:nodes), E.state_init_sd(sn, lay))
    rT = gr[1][2]
    @test Qo[rT[1], rT[2]] == sn.T0^2 && Qo[rT[1], rT[1]] == sn.T0^2 * (1 + sn.node_jitter^2) && Qo[rT[1], gr[2][2][1]] == 0
    @test Qn == Diagonal(Qn) && Qn[rT[1], rT[1]] == sn.T0^2 && isposdef(Qo) && isposdef(Qn)
    @test group_dofs(lay, sn)["T"] == 1 && group_dofs(lay, sn)["bark"] == 3 && group_dofs(lay, StateNoise(fields=:nodes))["T"] == lay.nx
    @test_throws ArgumentError state_noise_cov(lay, StateNoise(fields=:blob), E.state_init_sd(sn, lay))
    Pl = [2.0 0.3; 0.3 1.0]; Fl = [1.0 0.2; -0.1 0.8]; Ql = Diagonal([0.4, 0.2])
    @test E._sqrt_propagate(Fl, Pl, Ql) ≈ Fl * Pl * Fl' + Ql rtol=1e-10
    @test_throws ArgumentError E._stable_cov([1.0 0.0; 0.0 -1.0])
    # the sigma design: exactly zero mean and unit covariance, antithetic, reproducible
    Sg = sigma_design(17, 64, 7)
    @test size(Sg) == (64, 17) && maximum(abs.(sum(Sg, dims=1))) < 1e-10 && maximum(abs.(Sg' * Sg ./ 64 - I)) < 1e-10 && Sg == sigma_design(17, 64, 7)
    @test_throws ArgumentError sigma_design(17, 20, 7)
    # the state-noise file: the fingerprint is stable, the file round-trips, no file means the defaults
    fp = physics_fingerprint()
    @test length(fp) == 12 && fp == physics_fingerprint()
    tmp = tempname() * ".json"
    open(tmp, "w") do io
        JSON.print(io, merge(state_noise_dict(StateNoise(T=0.7, W0=0.2)), Dict("physics" => Dict("fingerprint" => fp))))
    end
    sn2 = load_state_noise(tmp; warn=false)
    @test sn2.T == 0.7 && sn2.W0 == 0.2 && sn2.C == sn.C && sn2.ignited0 == sn.ignited0 && sn2.fields == :offset
    @test load_state_noise(tempname() * ".json"; strict=false, warn=false) == StateNoise()
    # fail closed: a flight never runs on the defaults or on stale rates by accident; the tune round says so explicitly
    @test_throws ErrorException load_state_noise(tempname() * ".json")
    tmp_stale = tempname() * ".json"
    open(tmp_stale, "w") do io
        JSON.print(io, merge(state_noise_dict(StateNoise(T=0.7)), Dict("physics" => Dict("fingerprint" => "000000000000"))))
    end
    @test_throws ErrorException load_state_noise(tmp_stale)
    @test load_state_noise(tmp_stale; strict=false, warn=false).T == 0.7
    rm(tmp_stale)
    # the fingerprint covers every file the filter runs through: on a synthetic tree a byte in stream.jl or misfit.jl moves it
    @test "stream.jl" in E.FILTER_FILES && "misfit.jl" in E.FILTER_FILES
    mktempdir() do tree
        mkpath(joinpath(tree, "src")); mkpath(joinpath(tree, "estimator", "src"))
        write(joinpath(tree, "src", "a.jl"), "a")
        for f in E.FILTER_FILES; write(joinpath(tree, "estimator", "src", f), ""); end
        fp0 = physics_fingerprint(repo=tree)
        write(joinpath(tree, "estimator", "src", "stream.jl"), "b")
        fp1 = physics_fingerprint(repo=tree)
        write(joinpath(tree, "estimator", "src", "misfit.jl"), "c")
        fp2 = physics_fingerprint(repo=tree)
        @test fp0 != fp1 && fp1 != fp2 && length(fp2) == 12
    end
    rm(tmp)
end

@testset "the slice physics from the recipe's state is the whole-cook physics; the discrepancies do not enter it" begin
    keep = t .<= 1200.0 + 1e-9
    tw, uw = t[keep], u[keep]
    d = estimator_setup(tw, uw, pit, cfg, prior_phys, blk.phys_names)
    o0 = monolith_solve(x0[1:N_THETA], x0[(N_THETA + 1):N_PHYS], u[1:2], t[1:2], t[1:2]; pit, cfg, A_meat=exp(z0[AREA]), events...)
    ics, a = handoff(o0, 1)
    lay = FilterLayout(length(ics.T0f), length(sites), 4)
    v = vcat(z0, pack_state(ics, a, lay), [0.3, -0.2, 50.0, 0.01])
    ph = PhysicsState(AREA, events, lay, true)
    S1, lay1 = physics_stack(z0, d, Physics1D(area_latent=AREA, events=events))
    S2, lay2 = physics_stack(v, d, ph)
    nt, nx = lay1.nt, lay1.nx
    blocks = [(lay1.oTpit, nt, 0.05), (lay1.oT, nx * nt, 0.05), (lay1.oW, nx * nt, 1e-3), (lay1.opv, nt, 5.0), (lay1.oMass, nt, 1e-3)]
    @test lay2.nt == nt && all(maximum(abs.(S2[(o + 1):(o + len)] .- S1[(o + 1):(o + len)])) < tol for (o, len, tol) in blocks)
    S3, J3, _ = physics_outputs(v, d, ph)
    @test size(J3) == (length(S3), lay.n) && all(isfinite, J3) && all(J3[:, disc_range(lay)] .== 0)
    L = ph.last[]
    @test length(L.x_end) == lay.n_state && size(L.J_end) == (lay.n_state, lay.n) && all(isfinite, L.J_end) && all(L.J_end[:, disc_range(lay)] .== 0)
    @test abs(state_from_coord(L.x_end, lay)[5 * lay.nx + 4] - S3[lay2.oTpit + nt]) < 1e-6 # decoded end-state pit equals the stack
    @test maximum(abs.(L.J_end[:, state_range(lay)] .- Matrix(1.0 * I, lay.n_state, lay.n_state))) > 0.01   # the state moves the end state (not the identity)
    @test length(L.o.Tsurf) == nt
    # the channel wrapper on the interval: the discrepancy's column, white rows at the floor, the censored rows at the floor
    chs = channels()
    idxw = 1:nt
    kk = findlast(masks.pit[idxw]); keep1 = falses(nt); keep1[kk] = true
    sq = Sequential(restrict(window(chs[1], idxw), keep1), disc_index(lay, 1), white_floor(chs[1]))
    r, Jr = E.predict(sq, S3, J3, v, vcat(sites, state_sites(lay.n_state), disc_sites(4)), lay2)
    r0, _ = E.predict(restrict(window(chs[1], idxw), keep1), S3, J3, v, sites, lay2)
    @test maximum(abs.(r .- r0 .- v[disc_index(lay, 1)])) < 1e-12 && all(Jr[:, disc_index(lay, 1)] .== 1)
    W = E.whitening(sq)
    @test size(W) == (1, nt) && W[1, kk] == 1 / 0.1 && white_floor(sq) == 0.1
    kh = findfirst(k -> masks.humidity[k] && readings.humidity[k] >= nc.hum_cap - 1e-9, eachindex(t))
    idxh = (kh - 1):kh
    sh = Sequential(restrict(window(chs[3], idxh), [false, true]), disc_index(lay, 3), white_floor(chs[3]))
    Wh = E.whitening(sh)
    @test size(Wh) == (1, 2) && abs(Wh[1, 2] - 1 / 10.0) < 1e-12 && count(E.noise_mask(sh)) == 0     # a capped sample: one survival-site row at the floor, none in the residual chain
end

@testset "updates on arrival: the belief moves toward the readings, the diagnostics, the record" begin
    q = zeros(length(sites))
    recs = sequential_filter(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, diagnostics=true, t_end=1800.0)
    @test (nxr = (length(recs[end].x) - 9) ÷ 5; lr = FilterLayout(nxr, length(sites), 4);
           state_from_coord(recs[end].x, lr)[5 * nxr + 4] ≈ recs[end].T_pit[1] &&
           size(recs[end].x_cov) == (5 * nxr + 9, 5 * nxr + 9))
    @test filter_update!(filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q), 2).x === nothing
    # the sigma-point prediction against the tangent on a two-minute slice (nearly linear): the same means to a
    # fraction of a kelvin, widths within a factor, a positive-definite covariance, the latents' block identical
    st_s = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, n_sigma=64)
    st_l = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, n_sigma=0)
    rs = filter_update!(st_s, 2; diagnostics=true); rl = filter_update!(st_l, 2; diagnostics=true)
    @test rs.sigma && !rl.sigma && size(st_s.sigma) == (64, length(sites)) && size(st_l.sigma, 1) == 0
    @test abs(rs.T_pit[1] - rl.T_pit[1]) < 0.5 && abs(rs.T_centre[1] - rl.T_centre[1]) < 0.1 && abs(rs.T_surface[1] - rl.T_surface[1]) < 0.5
    @test 0.5 < rs.T_pit[2] / rl.T_pit[2] < 2 && 0.5 < rs.T_surface[2] / rl.T_surface[2] < 2
    @test isposdef(Symmetric(st_s.P)) && all(isfinite, st_s.P)
    # the statistical linearization against the tangent: with the belief's latent block shrunk to nothing the
    # regression is the tangent and the residual vanishes; with the belief as it is the residual is finite and
    # non-negative and the rows are scaled by it
    st0 = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, n_sigma=64)
    lay0 = st0.lay
    dd0 = estimator_setup(t[1:2], u[1:2], pit, cfg, sites[1:N_PHYS], names[1:N_PHYS])
    keep0 = trues(2)
    chs0 = E.Channel[E.Sequential(E.restrict(E.window(ch, 1:2), keep0), E.disc_index(lay0, c), E.white_floor(ch)) for (c, ch) in enumerate(st0.channels)]
    pw0 = FilterPrior(st0.joint, st0.m, inv(Symmetric(st0.P)))
    ph_t = PhysicsState(AREA, events, lay0, true)
    S_t, J_t, lay_t = physics_outputs(st0.m, dd0, ph_t)
    a_t, B_t, w_t = E.assemble(chs0, st0.m, S_t, J_t, pw0, lay_t)
    Ptiny = copy(st0.P); Ptiny[1:lay0.n_z, 1:lay0.n_z] .*= 1e-6
    ph_s = PhysicsState(AREA, events, lay0, true, st0.sigma, Ptiny)
    S_s, J_s, lay_s = physics_outputs(st0.m, dd0, ph_s)
    a_s, B_s, w_s = E.assemble_at(chs0, st0.m, S_s, J_s, pw0, lay_s, ph_s)
    @test maximum(abs.(a_s .- a_t)) < 1e-3 * maximum(abs.(a_t)) && isapprox(w_s, w_t; rtol=1e-6)   # the residual is the solver tolerance, not zero
    @test maximum(abs.(B_s[:, 1:lay0.n_z] .- B_t[:, 1:lay0.n_z])) < 1e-2 * maximum(abs.(B_t[:, 1:lay0.n_z]))
    @test maximum(abs.(B_s[:, (lay0.n_z + 1):end] .- B_t[:, (lay0.n_z + 1):end])) < 1e-5 * maximum(abs.(B_t[:, (lay0.n_z + 1):end]))
    ph_f = PhysicsState(AREA, events, lay0, true, st0.sigma, st0.P)
    S_f, J_f, lay_f = physics_outputs(st0.m, dd0, ph_f)
    a_f, B_f, w_f = E.assemble_at(chs0, st0.m, S_f, J_f, pw0, lay_f, ph_f)
    @test all(isfinite, a_f) && all(isfinite, B_f) && all(ph_f.scale[] .>= E.white_floor(chs0[1]) - 1e-12) && length(ph_f.scale[]) == length(a_f)
    arrivals = [k for k in 2:findlast(t .<= 1800.0 + 1e-9) if any(mk[k] for mk in (masks.pit, masks.core, masks.humidity, masks.mass))]
    # the filter's own first update (the initial readings assimilated, the prediction, the update at the first
    # arrival) under the damped posterior-linearization form: it commits and uses one to three rounds
    st_f = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, n_sigma=64)
    r_f = filter_update!(st_f, arrivals[1]; diagnostics=true)
    println("  the first update: ", r_f.rounds, " rounds, ", r_f.iters, " iterations, converged ", r_f.converged)
    @test r_f.updated && 1 <= r_f.rounds <= 3 && st_f.rounds == 3
    # the rounds' stopping rule: the KL between two Gaussians, zero for the same belief, half the squared shift for a shifted one
    @test E._gauss_kl(zeros(3), Matrix(1.0I, 3, 3), zeros(3), Matrix(1.0I, 3, 3)) == 0.0
    @test abs(E._gauss_kl(zeros(3), Matrix(1.0I, 3, 3), [0.3, 0.0, 0.4], Matrix(1.0I, 3, 3)) - 0.125) < 1e-12
    @test length(recs) == length(arrivals) && [r.k for r in recs] == arrivals && all(r.updated for r in recs)   # one update per arrival, none between
    r = recs[end]
    @test all(isfinite, r.z) && all(r.z_sd .> 0) && r.T_pit[2] > 0 && r.T_centre[2] > 0 && r.T_surface[2] > 0 && all(isfinite, r.disc)
    _, _, h0 = prior_terms(z0, sites)
    @test all(r.z_sd .<= sqrt.(1 ./ h0) .* (1 + 1e-6))                                  # no latent wider than its prior (no process noise on them)
    k = r.k
    @test abs(r.T_pit[1] - readings.pit[k]) < 6.0                                      # the belief's pit sits with the pit readings (the probe's noise is 1.5 K)
    @test isfinite(r.nis) && r.nis > 0 && r.dof > 0 && sum(length(v) for v in values(r.innov)) == r.dof
    @test size(r.T_cov) == (3, 3) && r.T_cov[1, 1] ≈ r.T_pit[2]^2
    @test length(r.noise) == 4 && length(r.level) == 4 && all(r.level .> 0) && !any(r.faulty)
    @test all(isfinite(x.misfit.total.z_chi2) for x in recs if x.misfit !== nothing)
    mis = innovation_misfit(recs; window_s=1800.0)
    @test mis !== nothing && isfinite(mis.z) && mis.dof == sum(x.dof for x in recs)
    # the diagnostics do not change the fit; a propagation without samples widens the state; a sample between updates is refused
    recs2 = sequential_filter(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, diagnostics=false, t_end=1800.0)
    @test maximum(abs.(recs2[end].z .- r.z)) < 1e-9
    st = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q)
    r1 = filter_update!(st, 2)
    for ch in st.channels; ch.mask[3:7] .= false; end                                  # the lid open: no samples for five steps
    r2 = filter_update!(st, 7)                                                          # a propagation to the seventh point, no update
    @test !r2.updated && st.k0 == 7 && st.past.t == t[7] && st.past.leth_s > 0
    xpost = state_from_coord(st.m[state_range(st.lay)], st.lay)
    kpost = E.state_kinds(st.lay)
    @test all(xpost[i] > 0 for i in eachindex(xpost) if kpost[i] == :positive)
    @test all(0 < xpost[i] < 1 for i in eachindex(xpost) if kpost[i] == :unit)
    @test isposdef(Symmetric(st.P))
    Lstate = cholesky(Symmetric(st.P[state_range(st.lay), state_range(st.lay)])).L
    rng_state = MersenneTwister(41)
    for _ in 1:20
        xd = state_from_coord(st.m[state_range(st.lay)] + Lstate * randn(rng_state, st.lay.n_state), st.lay)
        @test all(xd[i] > 0 for i in eachindex(xd) if kpost[i] == :positive)
        @test all(0 < xd[i] < 1 for i in eachindex(xd) if kpost[i] == :unit)
    end
    # the state's process noise: the same propagation without the rates is narrower (a damped state relaxes, so
    # the widths need not grow in time; they must exceed the rate-free ones)
    sn0 = StateNoise(T=0.0, W=0.0, C=0.0, F=0.0, S=0.0, bark=0.0, T_pit=0.0, m_fuel=0.0, m_pan=0.0, T_pan=0.0, T_fb=0.0, ignited=0.0)
    st0 = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, sn=sn0)
    filter_update!(st0, 2)
    for ch in st0.channels; ch.mask[3:7] .= false; end
    r20 = filter_update!(st0, 7)
    @test r2.T_pit[2] > r20.T_pit[2] && r2.T_centre[2] > r20.T_centre[2] && r2.T_surface[2] > r20.T_surface[2]
    for ch in st.channels; ch.mask[8] = true; end
    @test_throws ArgumentError filter_update!(st, 9)
    # an event's disturbance: the interval that crosses the add leaves the pit and the fuel wider by the event's
    # variances than the same interval without it (the transition through the chimney differs a little, so within a fifth)
    ev_add = (; events..., t_add=t[5])
    st_e = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q)
    st_a = filter_init(channels(), t, u, pit, cfg, sites, names, ev_add; noise, area_latent=AREA, q)
    for s in (st_e, st_a); filter_update!(s, 2); for ch in s.channels; ch.mask[3:7] .= false; end; end
    r_e = filter_update!(st_e, 7); r_a = filter_update!(st_a, 7)
    lay_e = st_e.lay
    Δpit_event = r_a.T_pit[2]^2 - r_e.T_pit[2]^2
    @test Δpit_event > 0
    @test abs(Δpit_event - st_e.sn.T_event^2) < 0.5 * st_e.sn.T_event^2 # nonlinear transition and coordinate map also move the variance
    _, Pa = state_physical_moments(st_a.m[state_range(lay_e)], st_a.P[state_range(lay_e), state_range(lay_e)], lay_e)
    _, Pe = state_physical_moments(st_e.m[state_range(lay_e)], st_e.P[state_range(lay_e), state_range(lay_e)], lay_e)
    ifuel = E.fuel_index(lay_e) - lay_e.n_z
    @test abs((Pa[ifuel, ifuel] - Pe[ifuel, ifuel]) - st_e.sn.m_add^2) < 0.2 * st_e.sn.m_add^2
    @test length(E._event_times(ev_add, t[end])) == length(E._event_times(events, t[end])) && any(te == t[5] && k == :add for (te, k) in E._event_times(ev_add, t[end]))
    @test load_state_noise(tempname() * ".json"; strict=false, warn=false).T_event == 5.0
    # endpoint process noise belongs to the predicted state before its reading: a precise pit sample after an
    # event conditions almost all of that event variance, while the same propagation without a sample retains it
    km = 7
    seen_mask = falses(length(t)); seen_mask[km] = true
    pit_seen = PitProbe(1.0, 0.0, seen_mask; data=readings.pit)
    pit_blind = PitProbe(1.0, 0.0, falses(length(t)); data=readings.pit)
    np_event = NoisePrior(1e-3, 1e-4, 0.0, 0.05)
    sn_event = StateNoise(T=0.0, W=0.0, C=0.0, F=0.0, S=0.0, bark=0.0, T_pit=0.0, m_fuel=0.0,
                          m_pan=0.0, T_pan=0.0, T_fb=0.0, ignited=0.0, T_event=8.0, T_spritz=0.0, m_add=0.0)
    st_seen = filter_init([pit_seen], t, u, pit, cfg, sites, names, ev_add; noise=[np_event], area_latent=AREA, q, sn=sn_event, n_sigma=0)
    st_blind = filter_init([pit_blind], t, u, pit, cfg, sites, names, ev_add; noise=[np_event], area_latent=AREA, q, sn=sn_event, n_sigma=0)
    r_seen = filter_update!(st_seen, km; diagnostics=true)
    r_blind = filter_update!(st_blind, km; diagnostics=true)
    @test r_seen.updated && !r_blind.updated
    @test r_seen.T_pit[2]^2 < 0.2 * r_blind.T_pit[2]^2
    @test abs(r_seen.T_pit[1] - readings.pit[km]) < 0.5
    # a failed endpoint fit is diagnostic only: the filter advances with the
    # predicted belief and does not adapt its residual/noise histories
    fail_mask = falses(length(t)); fail_mask[2] = true
    fail_ch = PitProbe(1.0, 0.0, fail_mask; data=readings.pit)
    predict_ch = PitProbe(1.0, 0.0, falses(length(t)); data=readings.pit)
    st_fail = filter_init([fail_ch], t, u, pit, cfg, sites, names, events; noise=[np_event], area_latent=AREA,
                          q, sn=sn0, n_sigma=0, iters=1, retry_iters=0, ftol=-1.0)
    st_predict = filter_init([predict_ch], t, u, pit, cfg, sites, names, events; noise=[np_event], area_latent=AREA,
                             q, sn=sn0, n_sigma=0, iters=1, retry_iters=0, ftol=-1.0)
    r_fail = filter_update!(st_fail, 2)
    r_predict = filter_update!(st_predict, 2)
    @test r_fail.attempted && !r_fail.updated && !r_fail.converged
    @test !r_predict.attempted && !r_predict.updated && r_predict.converged
    @test st_fail.m ≈ st_predict.m rtol = 1e-10 atol = 1e-10
    @test st_fail.P ≈ st_predict.P rtol = 1e-10 atol = 1e-10
    @test isempty(st_fail.dser[1]) && isempty(st_fail.eser[1]) && isempty(st_fail.didx[1])
    # the families' moments in z: the digamma and trigamma at known values, a lognormal's own, a beta's against draws
    @test abs(E._digamma(1.0) + 0.5772156649) < 1e-9 && abs(E._trigamma(1.0) - pi^2 / 6) < 1e-9 && abs(E._digamma(10.0) - 2.2517525890667) < 1e-9
    @test z_moments(PriorSite("lognormal", -3.2, 0.1)) == (-3.2, 0.1^2)
    pb = sites[names .== "a0"][1]                                     # the pilot's a0 in its score; the same family in the logit for the moments
    pl = PriorSite("beta", pb.p1, pb.p2)
    rngb = MersenneTwister(11)
    fd = [E._beta_rand(rngb, pb.p1, pb.p2) for _ in 1:200_000]
    zd = log.(fd ./ (1 .- fd))
    mb, vb = z_moments(pl)
    @test abs(mb - sum(zd) / length(zd)) < 0.02 && abs(sqrt(vb) - sqrt(sum((zd .- mb) .^ 2) / length(zd))) < 0.02
    zs = [to_z(f, pb) for f in fd]                                    # the same draws' scores: standard normal
    @test abs(sum(zs) / length(zs)) < 0.02 && abs(sqrt(sum(zs .^ 2) / length(zs)) - 1) < 0.02
    st_m = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q)
    ia = findfirst(==("a0"), names)
    @test st_m.m[ia] == 0.0 && st_m.P[ia, ia] == 1.0 && st_m.m[4] == sites[4].p1 && st_m.P[4, 4] == sites[4].p2^2
end

@testset "the plan from the filtered state is the whole cook from the start (the restart contract)" begin
    # the plant's own state at two hours, and the past's integrals over the two hours, stand in for the filter's
    o = monolith_solve(x0[1:N_THETA], x0[(N_THETA + 1):N_PHYS], u, t, t; pit, cfg, A_meat=exp(z0[AREA]), events...)
    k = findlast(t .<= 2 * 3600 + 1e-9)
    ics, a = handoff(o, k)
    lay = FilterLayout(length(ics.T0f), length(sites), 4)
    past = Past(); past.m0 = o.m_fuel[1]
    goals = PlanGoals()
    E._accumulate!(past, (Tsurf=o.Tsurf[1:k], Tcore=o.Tcore[1:k]), t[1:k], events.t_pull, goals)
    s = Schedule(t; u0=u[1]); s.u[1:k] = u[1:k]; s.t_now = t[k]                        # nothing committed yet: the plan decides the events
    d = estimator_setup(t, fill(0.15, length(t)), pit, cfg, prior_phys, blk.phys_names)
    f = free_set(s); mζ, _ = plan_prior(f, s)
    zz = copy(z0)                                                                       # the full block: the plan reads the physics sites and the area
    state = (k=k, t=t[k], ics=ics, ignited=a, x=pack_state(ics, a, lay), past=past, lay=lay, zx_cov=zeros(1, 1))
    y_state = plan_outputs_rh(zz, mζ, f, s, d; goals, state)
    y_whole = plan_outputs_rh(zz, mζ, f, s, d; goals)
    @test length(y_state) == length(y_whole) && all(isfinite, y_state)
    @test maximum(abs.(y_state[1:8] .- y_whole[1:8]) ./ (abs.(y_whole[1:8]) .+ 1e-3)) < 0.02        # the outcomes at serve: the hand-off's truncation
    @test maximum(abs.(y_state[(end - 1):end] .- y_whole[(end - 1):end]) ./ y_whole[(end - 1):end]) < 0.02   # the lethality integrals: past plus future
    o_s, p, _, tg = plan_solve(zz, mζ, f, s, d; state)
    @test tg[1] == t[k] && length(o_s.Tcore) == length(tg) && abs(o_s.Tcore[1] - o.Tcore[k]) < 1e-6
    o_w, _, _, _ = plan_solve(zz, mζ, f, s, d)
    @test maximum(abs.(o_s.Tcore .- o_w.Tcore[k:end])) < 0.1                            # the future from the state is the whole cook's future
    # the plan's predictive over the latents and the state
    V = zeros(length(zz) + lay.n_state, length(zz) + lay.n_state)
    for i in 1:length(zz); V[i, i] = 0.05^2; end
    for (j, i) in enumerate((length(zz) + 1):size(V, 1)); V[i, i] = (j <= lay.nx ? 1.0 : 1e-4)^2; end      # the temperature field a kelvin wide
    ics_decoded, a_decoded = unpack_state(state.x, lay)
    state_decoded = (state..., ics=ics_decoded, ignited=a_decoded)
    y_decoded = plan_outputs_rh(zz, mζ, f, s, d; goals, state=state_decoded)
    @test maximum(abs.(y_decoded .- y_state) ./ (abs.(y_state) .+ 1e-3)) < 0.02
    st2 = (state_decoded..., zx_cov=V)
    pr = plan_predictive(zz, V[1:length(zz), 1:length(zz)], mζ, f, s, d; goals, state=st2)
    @test length(pr.t) == length(tg) && pr.t[1] == t[k]
    @test all(isfinite, pr.Tcore.sd) && pr.Tcore.sd[1] > 0.5
    @test maximum(abs.(pr.y.mean .- y_decoded)) < 1e-9
end
println("test_sequential.jl: done")
