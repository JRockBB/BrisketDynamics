# The factor-graph backend's bench pieces: the accumulator's measurement groups against a simulation of their error
# dynamics on a small linear filter (a planted change of a sensor's white variance recovered), the bounded least
# squares and the Julia identification fit on the same statistics, the calibration file's round trip and its strict
# fingerprint; then on the real-cook mock's first arrivals the smoother's record carries the normalized innovation
# squared with its degrees of freedom and the standardized innovations, the accumulator is fed, and the process
# covariance's sensitivity to the state's variance rates sums to the mapped covariance where the cap is off.
#   julia --project=estimator estimator/test/test_bench_system.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, Statistics
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))

# a two-state linear plant read by two sensors, the filter at its steady-state gain: the accumulator's linear frame
function linear_filter()
    F = [0.9 0.1; 0.0 0.8]; Q = Matrix(Diagonal([0.04, 0.01])); H = Matrix(1.0I, 2, 2); Rw = Matrix(Diagonal([0.25, 1.0]))
    P = Matrix(1.0I, 2, 2)
    for _ in 1:500                                                             # the steady-state predicted covariance
        S = H * P * H' + Rw; K = P * H' / S
        P = F * (P - K * H * P) * F' + Q
    end
    Lw = Diagonal(1 ./ sqrt.(diag(Rw)))
    Hw = Lw * H; Sw = Hw * P * Hw' + I
    D = sqrt.(diag(Sw))
    Hs = Hw ./ D; Ss = Sw ./ (D * D'); K = P * Hs' / Ss
    return (F=F, Q=Q, Rw=Rw, P=P, Hs=Hs, Ss=Ss, K=K, D=D)
end

@testset "a sensor's white variance is a group: the accumulator's model against a simulation, the fit recovers it" begin
    lf = linear_filter()
    groups = ["q1", "q2", "white.1", "white.2"]
    θ = [lf.Q[1, 1], lf.Q[2, 2], lf.Rw[1, 1], lf.Rw[2, 2]]
    dQ = [[1.0 0; 0 0], [0.0 0; 0 1.0], zeros(2, 2), zeros(2, 2)]
    dR = [zeros(2, 2), zeros(2, 2), Matrix(Diagonal([1 / (lf.Rw[1, 1] * lf.D[1]^2), 0.0])), Matrix(Diagonal([0.0, 1 / (lf.Rw[2, 2] * lf.D[2]^2)]))]
    δθ = [0.0, 0.0, 0.5 * θ[3], 0.0]                                           # the first sensor's white variance half again as large
    Rrep = 3000; steps = 60
    rng = MersenneTwister(5)
    acc = ALSAccumulator(groups, θ, 2, 2; lag_s=10.0, bin_s=1.0)
    ε = zeros(2, Rrep)
    LS = cholesky(Symmetric(lf.Ss)).L
    for k in 1:steps
        ε = lf.F * ε                                                          # no excess process noise planted
        w = sqrt.(δθ[3] .* diag(dR[3])) .* randn(rng, 2, Rrep)               # the excess white noise, standardized
        e = lf.Hs * ε + LS * randn(rng, 2, Rrep) + w
        als_update!(acc, lf.F, dQ, lf.Hs, lf.Ss, lf.K, e, [1, 2], Float64(k); dR)
        ε = ε - lf.K * (lf.Hs * ε + w)                                        # the fixed gain acts on the whole excess innovation
    end
    @test acc.n_updates == steps && acc.n_rows > 0
    # the unbounded least squares on the groups the products pin
    σ2 = acc.rss / (acc.n_rows - 1)
    rel = [acc.G[g, g] > 0 && θ[g] > 0 ? sqrt(σ2 / acc.G[g, g]) / θ[g] : Inf for g in 1:4]
    a = findall(<(0.5), rel)
    @test 3 in a
    Ga = Symmetric(acc.G[a, a]); ha = acc.h[a]
    d = Ga \ ha
    rss1 = acc.rss - 2 * dot(ha, d) + dot(d, Ga * d)
    se = sqrt.(diag(inv(Ga)) .* (rss1 / (acc.n_rows - length(a))))
    println("  identified: ", join(["$(groups[g]) (rel se $(round(rel[g]; digits=2)))" for g in a], ", "))
    println("  white.1 planted +", round(δθ[3]; digits=4), ", fitted +", round(d[findfirst(==(3), a)]; digits=4), " ± ", round(se[findfirst(==(3), a)]; digits=4))
    @test all(abs(d[j] - δθ[a[j]]) <= 3 * se[j] for j in eachindex(a))
    # the same through the Julia fit on the pooled summary (a JSON round trip, as the driver reads it): the group identified,
    # the change recovered within three standard errors, the rest held near their numbers
    s = JSON.parse(JSON.json(als_summary(acc)))
    pooled = als_pool([s])
    @test pooled.groups == groups && pooled.n_rows == acc.n_rows && length(pooled.blocks) > 0
    fit = als_fit(pooled; cap=4.0)
    @test fit["identified"][3]
    @test abs(fit["theta_new"][3] - θ[3] - δθ[3]) <= 3 * max(2 * fit["rates_new"][3] * fit["se_rate"][3], 1e-9)
    @test all(abs(fit["theta_new"][g] - θ[g]) <= 3 * (fit["se_rate"][g] === nothing ? 0.0 : 2 * fit["rates_new"][g] * fit["se_rate"][g]) + 1e-9 for g in (1, 2, 4) if fit["identified"][g])
    @test fit["rss_after"] < fit["rss_before"]
    println(format_als_fit(fit, Dict("q1" => "-", "q2" => "-", "white.1" => "-", "white.2" => "-"), Dict(1 => "one", 2 => "two")))
    rows = als_lag_table(pooled, fit, Dict(1 => "one", 2 => "two"), [0, 1, 2, 5])
    @test !isempty(rows) && all(r["n"] >= 2 for r in rows)
    println(format_als_lags(rows))
    # three cooks' summaries give a jackknife
    fit3 = als_fit(als_pool([s, s, s]); cap=4.0, summaries=[s, s, s])
    @test fit3["jackknife"] !== nothing && fit3["jackknife"]["n_cooks"] == 3 && all(v == 0 || v === nothing || v < 1e-9 for v in fit3["jackknife"]["se_rate"])
end

@testset "the bounded least squares: the KKT conditions hold, the cap binds where it must" begin
    rng = MersenneTwister(2)
    for trial in 1:20
        n = rand(rng, 2:6)
        A = randn(rng, n + 3, n); G = A' * A + 0.1I; h = randn(rng, n) .* 3
        lo = -rand(rng, n) .* 0.5; hi = rand(rng, n) .* 0.5
        x = E._bounded_ls(G, h, lo, hi)
        @test all(lo .- 1e-12 .<= x .<= hi .+ 1e-12)
        grad = G * x .- h
        for i in 1:n
            if x[i] <= lo[i] + 1e-9
                @test grad[i] >= -1e-8
            elseif x[i] >= hi[i] - 1e-9
                @test grad[i] <= 1e-8
            else
                @test abs(grad[i]) < 1e-8
            end
        end
        cost(v) = dot(v, G * v) - 2 * dot(h, v)
        @test all(cost(x) <= cost(clamp.(x .+ 0.1 .* randn(rng, n), lo, hi)) + 1e-9 for _ in 1:20)
    end
    x0 = E._bounded_ls([2.0 0; 0 2.0], [10.0, -10.0], [-1.0, -1.0], [1.0, 1.0])
    @test x0 ≈ [1.0, -1.0]
    xf = E._bounded_ls([2.0 0.5; 0.5 2.0], [0.2, -0.1], [-1.0, -1.0], [1.0, 1.0])
    @test xf ≈ [2.0 0.5; 0.5 2.0] \ [0.2, -0.1]
end

@testset "the calibration file: a round trip, and a stale backend fingerprint is refused" begin
    sn = StateNoise(T=0.3, T_pit=1.1)
    sensors = Dict(:pit => (white_sd=0.1, ar_sd=1.4, tau=3000.0), :core => (white_sd=0.1, ar_sd=1.1, tau=12000.0))
    bl_one = belief_layout(CookSystem(Slab1D(:x; cfg=MeatConfig(; nx=4)); pit=PitConfig()))
    noise = state_noise_record(bl_one, sn)
    @test Set(keys(noise)) == Set(vcat([string(k) for k in fieldnames(StateNoise)], ["field_modes"])) && noise["T"] == 0.3 && noise["T0"] == 1.0 && noise["fields"] == "offset" && noise["field_modes"] == E.FIELD_MODES
    rec = system_calibration_record(noise, sensors; method="a test")
    @test rec["backend"] == "system" && rec["physics"]["fingerprint"] == backend_fingerprint() && rec["rates"]["T"] == 0.3 && rec["units"]["T"] == "K"
    # two pieces: the groups carry the piece's name, the record gives a StateNoise per piece with the pit's shared
    bl_two = belief_layout(CookSystem([Slab1D(:a; cfg=MeatConfig(; nx=4)), Slab1D(:b; cfg=MeatConfig(; nx=4))]; pit=PitConfig()))
    noise2 = state_noise_record(bl_two, Dict(:a => StateNoise(T=0.7), :b => StateNoise(W=0.02), :pit => StateNoise(T_pit=3.0)))
    @test noise2["a.T"] == 0.7 && noise2["b.T"] == 0.5 && noise2["b.W"] == 0.02 && noise2["T_pit"] == 3.0 && noise2["a.T0"] == 1.0
    rec2 = system_calibration_record(noise2, sensors; method="two pieces")
    sn2d = system_state_noise(rec2)
    @test sn2d isa Dict && sn2d[:a].T == 0.7 && sn2d[:b].W == 0.02 && sn2d[:a].T_pit == 3.0 && sn2d[:pit].T_pit == 3.0 && sn2d[:b].T == 0.5
    @test maximum(abs.(belief_process_cov(bl_two, sn2d, 1.0) .- belief_process_cov(bl_two, Dict(:a => StateNoise(T=0.7, T_pit=3.0), :b => StateNoise(W=0.02, T_pit=3.0), :pit => StateNoise(T_pit=3.0)), 1.0))) < 1e-12
    path = tempname() * ".json"
    write_system_calibration(path, rec)
    sn2, sensors2 = load_system_calibration(path; strict=true)
    @test sn2.T == 0.3 && sn2.T_pit == 1.1 && sn2.W == StateNoise().W
    @test sensors2[:pit].ar_sd == 1.4 && sensors2[:core].tau == 12000.0
    d = JSON.parsefile(path); d["physics"]["fingerprint"] = "000000000000"
    open(path, "w") do io; JSON.print(io, d, 1); end
    @test_throws ErrorException load_system_calibration(path; strict=true)
    sn3, _ = load_system_calibration(path; strict=false, warn=false)
    @test sn3.T == 0.3
    rm(path)
    @test_throws ErrorException load_system_calibration(tempname() * ".json"; strict=true)
    @test load_system_calibration(tempname() * ".json"; strict=false, warn=false) === (nothing, nothing)
end

# --- on the physics: the real-cook mock's first arrivals -----------------------------------------------------------------
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
ev = m["events"]
events = [CookEvent(:wrap, :brisket, ev["t_wrap"]), CookEvent(:pull, :brisket, ev["t_pull"]), CookEvent(:add, :pit, ev["t_add"]),
          CookEvent(:spritz, :brisket, ev["t_spritz"]; dt=ev["dt_spritz"], m=ev["m_spritz"])]
nc = RealCookNoise()
sys = CookSystem(Slab1D(:brisket; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[:brisket])
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=nc.pit_sd, tau=nc.pit_tau)),
           SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=nc.core_sd, tau=nc.core_tau)),
           SensorSpec(:humidity, ChamberVapor(), NoiseModel(white_sd=10.0, ar_sd=nc.hum_sd, tau=nc.hum_tau, cap=nc.hum_cap)),
           SensorSpec(:mass, TotalMass(), NoiseModel(white_sd=0.005, ar_sd=nc.mass_sd, tau=nc.mass_tau))]
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
K_END = 3
recs = MeasurementRecord[]
for k in 1:K_END, id in (:pit, :core, :humidity, :mass)
    getfield(masks, id)[k] && push!(recs, MeasurementRecord(id, t[k], getfield(readings, id)[k]))
end
arrivals = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
clog = CookLog(t, u; events, sensors, measurements=recs)
model = SystemModel(sys, clog, blk; sn=StateNoise())

@testset "the smoother's record carries the predicted-covariance consistency; the accumulator is fed in the newest-node frame" begin
    feed = ALSFeed(model; lag_s=3600.0, bin_s=dt)
    @test feed.acc.groups[1:12] == collect(STATE_GROUPS) && feed.acc.groups[13:16] == ["ou.pit", "ou.core", "ou.humidity", "ou.mass"]
    @test feed.acc.groups[17:20] == ["white.pit", "white.core", "white.humidity", "white.mass"] && feed.acc.n == 17 + model.bl.n + 4
    @test feed.acc.theta[1] == StateNoise().T^2 && feed.acc.theta[13] == nc.pit_sd^2 && feed.acc.theta[17] == 0.1^2
    sm = online_init(model; lag=1, als=feed)
    out = [assimilate!(sm, grp) for grp in arrivals]
    @test all(r.converged && r.committed for r in out)
    for (r, grp) in zip(out, arrivals)
        live = [x for x in grp if !(x.sensor === :humidity && x.value >= nc.hum_cap - 1e-9)]
        @test r.dof == length(live) && isfinite(r.nis) && r.nis >= 0
        @test Set(keys(r.innov_std)) == Set(x.sensor for x in live) && all(isfinite, values(r.innov_std))
    end
    # the graph's own residuals at the accepted point, and the smoothed past as a node leaves (lag 1: from the third arrival)
    rs = out[end].resid
    @test rs !== nothing && Set(keys(rs.dyn)) == Set(STATE_GROUPS) && Set(keys(rs.dyn_window)) == Set(STATE_GROUPS)
    @test all(v[2] == length(model.bl.groups[findfirst(x -> E._als_group_name(model.bl, first(x)) == g, model.bl.groups)][2]) for (g, v) in rs.dyn)
    @test all(rs.dyn_window[g][2] == 2 * rs.dyn[g][2] && rs.dyn_window[g][1] >= rs.dyn[g][1] for g in keys(rs.dyn))   # two slices in the window before the oldest leaves at lag 1
    @test all(v[1] >= 0 for v in values(rs.dyn)) && rs.rows > rs.cols > 0
    @test Set(keys(rs.obs)) == Set(x.sensor for x in arrivals[end] if !(x.sensor === :humidity && x.value >= nc.hum_cap - 1e-9)) && all(v >= 0 for v in values(rs.obs))
    @test all(haskey(rs.ou, id) for id in (:pit, :core)) && all(v >= 0 for v in values(rs.ou))
    @test isempty(out[1].resid.dyn) && out[1].past === nothing && out[2].past === nothing              # the first arrival is the initial node; the window fills before a node leaves
    @test out[3].past !== nothing && out[3].past.t == t[1] && length(out[3].past.y) == model.bl.n && size(out[3].past.y_cov) == (model.bl.n, model.bl.n)
    @test feed.acc.n_updates == length(out) && feed.acc.n_rows > 0 && issymmetric(feed.acc.G)
    @test minimum(eigvals(Symmetric(feed.acc.G))) >= -1e-9 * maximum(abs.(feed.acc.G))
    # the accumulator is a reading, not a change: the beliefs equal those of a smoother without it
    sm0 = online_init(model; lag=1)
    out0 = [assimilate!(sm0, grp) for grp in arrivals]
    @test out0[end].z ≈ out[end].z && out0[end].y ≈ out[end].y && out0[end].nis ≈ out[end].nis
    # lag 0: the previous node is re-solved once with the new readings before it leaves, so every arrival after the first carries
    # that node's belief as its past; the newest slice's residual is read before it is marginalized, and it is the window's whole
    sm00 = online_init(model; lag=0)
    out00 = [assimilate!(sm00, grp) for grp in arrivals]
    @test out00[1].past === nothing && all(r.past !== nothing for r in out00[2:end]) && out00[end].past.t == t[K_END - 1]
    @test out00[end].resid !== nothing && !isempty(out00[end].resid.dyn) && out00[end].resid.dyn == out00[end].resid.dyn_window
    # the summary pools and the driver's fit runs on it (too few rows to identify anything: every group held)
    s = JSON.parse(JSON.json(als_summary(feed.acc)))
    pooled = als_pool([s])
    @test pooled.n_rows == feed.acc.n_rows && length(pooled.groups) == 20
    # the process covariance's sensitivity to the state's variance rates sums to the mapped covariance less the floor where the cap is off
    kp, ky = sm.nodes[end - 1][2], sm.nodes[end][2]
    f = sm.dyn[(kp, ky)]
    z = E.variable_value(sm.graph, sm.key_z); y0 = E.variable_value(sm.graph, kp)
    c = E._dyn_eval!(f, z, y0; jac=true)
    dQ = E._process_cov_sens(f, c.f, feed.state_names)
    θ = feed.acc.theta[1:12]
    Qsum = sum(θ[g] .* dQ[g] for g in 1:12)
    Δh = (f.t1 - f.t0) / 3600
    Qphys = belief_process_cov(model.bl, model.sn, Δh; events=f.events)
    Qc = E._process_cov_coord(f, c.f, Qphys); Qc0 = E._process_cov_coord(f, c.f, zeros(size(Qphys)))
    d = inv.(max.(belief_decode_jacobian(c.f, model.bl), eps(Float64)))
    sd = sqrt.(diag(Qphys) .* d .^ 2)
    free = findall(sd .< 0.05 * 3.0)
    @test length(free) > 10
    @test maximum(abs.(Qsum[free, free] .- (Qc .- Qc0)[free, free])) <= 1e-8 * maximum(abs.(Qc))
    @test all(size(D) == (model.bl.n, model.bl.n) && issymmetric(D) for D in dQ)
end
@testset "drifting latents: a copy per node tied by the random walk; tied tightly it is the static smoother" begin
    idx = Dict(nm => i for (i, nm) in enumerate(blk.names))
    moving = [idx["u_leak"], idx["k_e"], idx["h_c"], idx["r_max"]]
    # tied at the floor (a walk of 1e-3 per √h): the beliefs are the static smoother's to a fraction of a percent
    tight = (moving=moving, sd_per_hour=zeros(4), jumps=Tuple{Int,Float64,Float64}[])
    sm_s = online_init(model; lag=1)
    sm_d = online_init(model; lag=1, drift=tight)
    @test sm_d.split !== nothing && sm_d.split.moving == moving && length(sm_d.split.static) == 13
    @test length(variable_value(sm_d.graph, key_params(sm_d))) == 13 && length(variable_value(sm_d.graph, key_drift(sm_d, 1))) == 4
    out_s = [assimilate!(sm_s, grp) for grp in arrivals]
    out_d = [assimilate!(sm_d, grp) for grp in arrivals]
    @test all(r.converged && r.committed for r in out_d)
    for (a, b) in zip(out_s, out_d)
        @test length(b.z) == 17 && maximum(abs.(a.z .- b.z) ./ sqrt.(diag(a.z_cov))) < 0.02
        @test maximum(abs.(sqrt.(diag(b.z_cov)) ./ sqrt.(diag(a.z_cov)) .- 1)) < 0.02
        @test abs(a.T_pit[1] - b.T_pit[1]) < 0.02 && abs(a.pieces[:brisket].T_centre[1] - b.pieces[:brisket].T_centre[1]) < 0.01
        @test abs(a.nis - b.nis) < 0.05 * max(a.nis, 1.0)
    end
    # the window's structure: a copy per node, a random-walk factor per slice, the oldest copy marginalized with its node
    ks = [k for (_, k) in sm_d.nodes]
    @test length(ks) == 2 && all(has_variable(sm_d.graph, key_drift(sm_d, k.k)) for k in ks)
    @test length(sm_d.drift_fac) == 1 && haskey(sm_d.drift_fac, (key_drift(sm_d, ks[1].k), key_drift(sm_d, ks[2].k)))
    @test !has_variable(sm_d.graph, key_drift(sm_d, 1)) && count(f -> f isa HierarchyFactor, sm_d.graph.factors) == 1
    # the random walk's widths: the sd between copies over a slice is the rate over its hours, a jump adds its variance on the slice crossing it
    walk = (moving=moving, sd_per_hour=[0.05, 0.05, 0.03, 0.0], jumps=[(idx["r_max"], t[2] + 30.0, 0.15)])
    sm_w = online_init(model; lag=2, drift=walk)
    for grp in arrivals[1:3]; assimilate!(sm_w, grp); end
    kw = [k for (_, k) in sm_w.nodes]
    f12 = sm_w.drift_fac[(key_drift(sm_w, kw[1].k), key_drift(sm_w, kw[2].k))]
    f23 = sm_w.drift_fac[(key_drift(sm_w, kw[2].k), key_drift(sm_w, kw[3].k))]
    Δh = dt / 3600
    @test f12.τ[1] ≈ 0.05 * sqrt(Δh) && f12.τ[3] ≈ 0.03 * sqrt(Δh) && f12.τ[4] ≈ sqrt(E.DRIFT_Q_FLOOR * Δh)
    @test f23.τ[4] ≈ sqrt(E.DRIFT_Q_FLOOR * Δh + 0.15^2) && f23.τ[1] ≈ 0.05 * sqrt(Δh)
    b = belief_at(sm_w)
    @test length(b.z) == 17 && size(b.zy_cov) == (17 + model.bl.n, 17 + model.bl.n) && all(isfinite, b.z)
    @test_throws ArgumentError online_init(model; drift=(moving=[1, 1], sd_per_hour=[0.1, 0.1], jumps=Tuple{Int,Float64,Float64}[]))
    @test_throws ArgumentError online_init(model; drift=(moving=[1], sd_per_hour=[0.1], jumps=[(2, 100.0, 0.1)]))
    @test_throws ArgumentError online_init(model; drift=(moving=collect(1:17), sd_per_hour=fill(0.1, 17), jumps=Tuple{Int,Float64,Float64}[]))
end
@testset "the fields' noise: the bulk, the smooth modes and the jitter; the NEES on the mode coefficients" begin
    # the basis: orthogonal columns of squared norm n, the bulk first; fewer when the nodes cannot carry them
    for (n, M) in ((10, 2), (12, 3), (2, 2), (1, 2))
        B = E.field_mode_basis(n, M)
        @test size(B, 1) == n && size(B, 2) == min(M + 1, n)
        @test all(B[:, 1] .== 1.0)
        @test maximum(abs.(B' * B - n * I)) < 1e-10
    end
    # the count ships at zero (the bulk alone: with two modes the probe's depth drifted to 6.5 sd in one cook, system_state.jl);
    # the machinery is exercised with the count set on the layout
    @test E.FIELD_MODES == 0
    sys1 = CookSystem(Slab1D(:b; cfg=MeatConfig(; nx=10)); pit=PitConfig())
    bl1 = belief_layout(sys1); sn1 = StateNoise()
    @test bl1.modes[:b] == 0 && size(E.field_mode_basis(bl1, "b.T"), 2) == 1
    Q0 = belief_process_cov(bl1, sn1, 1.0)
    rT = bl1.groups[findfirst(p -> first(p) == "b.T", bl1.groups)][2]
    @test maximum(abs.(Q0[rT, rT] .- sn1.T^2 .* (ones(10, 10) + sn1.node_jitter^2 * I))) < 1e-14      # the bulk and the jitter, as before
    bl1.modes[:b] = 2
    # a one-slab system with two modes: the T group's slice covariance is s² (B Bᵀ + jitter² I); the scalars and the bark untouched
    @test bl1.modes[:b] == 2 && E.field_mode_basis(bl1, "T_pit") === nothing && E.field_mode_basis(bl1, "b.bark") === nothing
    Q = belief_process_cov(bl1, sn1, 1.0)
    B = E.field_mode_basis(bl1, "b.T")
    @test size(B) == (10, 3)
    @test maximum(abs.(Q[rT, rT] .- sn1.T^2 .* (B * B' + sn1.node_jitter^2 * I))) < 1e-12
    ip = first(E.belief_index(bl1, :pit, :T_pit))
    @test Q[ip, ip] ≈ sn1.T_pit^2 && all(Q[ip, j] == 0 for j in rT)
    # what the whitener charges: a bulk shift of 1 K and a linear tilt of RMS 1 K cost the same; bending one node by
    # 1 K is priced at the jitter, so a probe still cannot bend the field to its own noise
    Qt = Matrix(Symmetric(Q[rT, rT]))
    cost(e) = dot(e, Qt \ e)
    bulk = ones(10); tilt = B[:, 2]; bend = zeros(10); bend[5] = 1.0
    @test isapprox(cost(bulk), cost(tilt); rtol=1e-8)
    @test cost(bend) > 1000 * cost(tilt)
    @test cost(bend) > 0.5 / (sn1.T^2 * sn1.node_jitter^2)
    # the variance form agrees with the covariance form (the accumulator's derivative goes through it)
    Qv = E._belief_process_var(bl1, sn1, g -> E._group_rate(bl1, sn1, g, false)^2)
    @test maximum(abs.(Qv .- Q)) < 1e-12
    # the record and the file carry the mode count
    rec = state_noise_record(bl1, sn1)
    @test rec["field_modes"] == E.FIELD_MODES
    cal = E.system_calibration_record(rec, Dict(:pit => (white_sd=0.1, ar_sd=1.5, tau=3000.0)))
    @test cal["field_modes"] == E.FIELD_MODES
    # the NEES on the mode coefficients: a field error of one sd in every coefficient scores one per degree of freedom
    A = B' ./ 10
    e = B * [1.0, -1.0, 1.0] .* sn1.T
    Ca = A * Qt * A'
    @test isapprox(dot(A * e, Ca \ (A * e)) / 3, 1.0; rtol=1e-6)
end
println("test_bench_system.jl: done")
