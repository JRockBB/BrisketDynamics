# The offline backend (offline.jl): the full history of two cooks on one graph with a hierarchy, on a small
# mesh and a short horizon, against planted truths:
#   multiple shooting  every node's belief on a cook read by its own pit and core probes converges from the prior
#   hierarchy          the convective coefficient's population value and the two cooks' offsets recovered within
#                      their posterior widths; the pooled widths no wider than one cook's alone
#   unobservable       graph-information loss names disconnected groups without mistaking indirect coupling
#   sparse = dense     the sparse solve and marginals equal the dense ones
#   julia --project=estimator estimator/test/test_offline.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
prior_file = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
prior_phys = read_sites(prior_file["prior"]); phys_names = String.(prior_file["names"])
cfg = MeatConfig(; nx=10, T0=299.8)
pit = PitConfig(; T_amb=300.0, Tpit0=362.4, m0=7.0)
HR = 3600.0
t = collect(0.0:600.0:(2400.0 * 2))                             # a 80-minute cook on a ten-minute grid, nine points
u = fill(0.25, length(t))
sys = CookSystem(Slab1D(:b; cfg); pit)
blk = system_latents(sys, prior_phys, phys_names)                 # no probe-depth latent: the core probe at the centre
I_HC = latent_index(blk, "h_c")
z0 = [z_moments(p)[1] for p in blk.sites]
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.5)),
           SensorSpec(:core, TemperatureAt(:b; s=0.0), NoiseModel(white_sd=0.3))]
τ_hc = 0.15
z_pop = copy(z0); z_pop[I_HC] += 0.25                             # the population's h_c a quarter above the prior's centre
rng = MersenneTwister(21)
offsets = [0.12, -0.10]
function cook_log(zc, seed)
    r = MersenneTwister(seed)
    x = from_z(zc[1:N_PHYS], blk.sites[1:N_PHYS])
    o = system_solve(sys, [x[1:N_THETA]], [exp(zc[latent_index(blk, "A_meat")])], x[(N_THETA + 1):N_PHYS], u, t, t)
    recs = MeasurementRecord[]
    for k in eachindex(t)
        push!(recs, MeasurementRecord(:pit, t[k], o.Tpit[k] + 0.5 * randn(r)))
        push!(recs, MeasurementRecord(:core, t[k], o.piece[:b].Tcore[k] + 0.3 * randn(r)))
    end
    return CookLog(t, u; sensors, measurements=recs), o
end
truths = [(zc = copy(z_pop); zc[I_HC] += off; zc) for off in offsets]
logs = [cook_log(truths[i], 100 + i) for i in 1:2]
models = [SystemModel(sys, logs[i][1], blk; sn=StateNoise()) for i in 1:2]
const ONLY = filter(!isempty, split(get(ENV, "TEST_OFFLINE_ONLY", ""), ","))
run_set(name) = isempty(ONLY) || name in ONLY

run_set("legacy") && @testset "two cooks, one graph: the full histories converge, the hierarchy pools the shared latent" begin
    t0 = time()
    prob = offline_problem(models; ids=[:a, :b], hierarchy=[I_HC], tau=[τ_hc], iters=25, tol=1e-5, ftol=1e-6)
    @test length(prob.cooks) == 2 && prob.key_pop == VarKey(:hyper, :population, 0)
    @test all(length(sm.nodes) == length(t) for sm in prob.cooks) && all(sm.lag == typemax(Int) for sm in prob.cooks)
    @test prob.graph.keys[end] == prob.key_pop && prob.graph.keys[end - 1] == prob.cooks[2].key_z    # the parameters last
    r = solve!(prob; sparse=true, verbose=false)
    println("  the offline solve: ", length(r.hist), " iterations, converged ", r.converged,
            ", scaled gradient ", r.gradient, ", ", round(Int, time() - t0), " s")
    r.converged || println("  final iterations: ", r.hist[max(1, end - 4):end])
    @test r.converged
    pop = population(prob)
    sp = sqrt(pop.cov[1, 1])
    @printf("  population h_c: %.3f ± %.3f (planted %.3f); cooks: %s\n", pop.mean[1], sp, z_pop[I_HC],
            join([@sprintf("%.3f ± %.3f (truth %.3f)", cook_latents(prob, i).mean[I_HC], sqrt(cook_latents(prob, i).cov[I_HC, I_HC]), truths[i][I_HC]) for i in 1:2], "; "))
    @test abs(pop.mean[1] - z_pop[I_HC]) < 3 * sp
    for i in 1:2
        cl = cook_latents(prob, i)
        sc = sqrt(cl.cov[I_HC, I_HC])
        @test abs(cl.mean[I_HC] - truths[i][I_HC]) < 3 * sc
        @test sc < sqrt(z_moments(blk.sites[I_HC])[2])                       # narrower than the prior
        b = cook_belief(prob, i)
        @test abs(b.T_pit[1] - logs[i][2].Tpit[end]) < 2.0 && abs(b.pieces[:b].T_centre[1] - logs[i][2].piece[:b].Tcore[end]) < 1.0
        @test all(isfinite, b.x) && b.T_pit[2] > 0
        # Graph information, not direct row incidence: the inert salt and unexcited pan/firebox states remain
        # unobserved, while collagen, fat and bark gain indirect information through the thermal dynamics.
        ug = unobserved_groups(prob, i)
        println("  cook ", i, " unobserved groups: ", join(ug, ", "))
        @test all(g in ug for g in ("b.S", "m_pan", "T_pan", "T_fb"))
        @test !any(g in ug for g in ("b.T", "T_pit", "b.C", "b.F", "b.bark"))
    end
    # the sparse and the dense marginals agree
    Cd = marginal_cov(prob.graph, prob.key_pop; sparse=false); Cs = marginal_cov(prob.graph, prob.key_pop; sparse=true)
    @test maximum(abs.(Cd .- Cs)) < 1e-8 * max(1, maximum(abs.(Cd)))
    # without the hierarchy each cook stands alone: the pooled posterior on h_c is no wider than either alone
    prob1 = offline_problem(models[1:1]; ids=[:a], iters=25, tol=1e-5, ftol=1e-6)
    r1 = solve!(prob1; sparse=true)
    @test r1.converged && prob1.key_pop === nothing
    s1 = sqrt(cook_latents(prob1, 1).cov[I_HC, I_HC])
    @test sqrt(cook_latents(prob, 1).cov[I_HC, I_HC]) <= s1 * (1 + 1e-6)
    @test_throws ArgumentError offline_problem(models; ids=[:a, :a])
    @test_throws ArgumentError offline_problem(models; hierarchy=[I_HC], tau=Float64[])
end

run_set("semantic") && @testset "semantic multilevel hierarchy supports heterogeneous layouts and sensor suites" begin
    # Cook b carries an additional sensor-scoped depth latent and only a core sensor. Membership is by
    # LatentKey, not by vector position; its family parent is itself tied to a population parent.
    blk_b = system_latents(sys, prior_phys, phys_names; probes=[:b])
    core_sensor = only(filter(s -> s.id === :core, sensors))
    core_records = filter(r -> r.sensor === :core, collect(logs[2][1].measurements))
    log_b = CookLog(t, u; sensors=[core_sensor], measurements=core_records)
    hetero_models = [models[1], SystemModel(sys, log_b, blk_b; sn=StateNoise())]
    child_a = latent_key(blk, I_HC)
    child_b = latent_key(blk_b, latent_index(blk_b, "h_c"))
    family = LatentKey(:family, :brisket, :h_c)
    population_key = LatentKey(:population, :all_meat, :h_c)
    hs = HierarchySpec(family, Dict(:a => child_a, :b => child_b), τ_hc;
                       ancestor=population_key, ancestor_tau=0.30)
    prob = offline_problem(hetero_models; ids=[:a, :b], hierarchies=[hs],
                           iters=25, tol=2e-5, ftol=1e-6)
    @test length(blk_b) == length(blk) + 1
    @test length(hetero_models[1].log.sensors) == 2 && length(hetero_models[2].log.sensors) == 1
    @test haskey(prob.hyper, family) && haskey(prob.hyper, population_key)
    @test any(f -> f isa HierarchyFactor && f.child == prob.hyper[family] &&
                   f.parent == prob.hyper[population_key], prob.graph.factors)
    r = solve!(prob; sparse=true)
    println("  semantic hierarchy solve: ", length(r.hist), " iterations, converged ", r.converged,
            ", scaled gradient ", r.gradient)
    r.converged || println("  final iterations: ", r.hist[max(1, end - 4):end])
    @test r.converged
    fam = hyperposterior(prob, family)
    @test abs(fam.mean[1] - z_pop[I_HC]) < 3 * sqrt(fam.cov[1, 1])
    @test cook_latents(prob, 1).keys[I_HC] == child_a
    @test cook_latents(prob, 2).keys[latent_index(blk_b, "h_c")] == child_b
    logit_a = PriorSite("beta", 2.0, 3.0, 0.0, 1.0; coord=:logit)
    logit_b = PriorSite("beta", 4.0, 5.0, 0.0, 1.0; coord=:logit)
    score_a = PriorSite("beta", 2.0, 3.0, 0.0, 1.0; coord=:score)
    score_b = PriorSite("beta", 4.0, 5.0, 0.0, 1.0; coord=:score)
    @test E._compatible_site(logit_a, logit_b)                       # shape changes the prior, not the logit transform
    @test !E._same_site_prior(logit_a, logit_b)                      # a shared root cannot inherit either one by order
    @test !E._compatible_site(score_a, score_b)                      # score transforms depend on the beta shape
    @test !E._compatible_site(PriorSite("logbeta", 2.0, 3.0, -2.0, 1.0; coord=:logit),
                              PriorSite("logbeta", 2.0, 3.0, -1.0, 1.0; coord=:logit))
    duplicate_child = HierarchySpec(LatentKey(:family, :duplicate, :h_c),
                                    Dict(:a => child_a), τ_hc)
    @test_throws ArgumentError offline_problem(hetero_models; ids=[:a, :b],
                                               hierarchies=[hs, duplicate_child])
    @test_throws ArgumentError HierarchySpec(family, Dict(:a => child_a), NaN)
    @test_throws ArgumentError HierarchySpec(family, Dict(:a => child_a), τ_hc;
                                             ancestor=LatentKey(:piece, :bad, :h_c), ancestor_tau=0.2)
    @test_throws ArgumentError HierarchySpec(family, Dict(:a => LatentKey(:piece, :b, :D_eff)), τ_hc)

    # One cook may contribute several piece instances to the same family parent.
    sys_two = CookSystem([Slab1D(:left; cfg), Slab1D(:right; cfg)]; pit)
    blk_two = system_latents(sys_two, prior_phys, phys_names)
    model_two = SystemModel(sys_two, CookLog(t, u), blk_two; sn=StateNoise())
    left_hc = latent_key(blk_two, latent_index(blk_two, "left.h_c"))
    right_hc = latent_key(blk_two, latent_index(blk_two, "right.h_c"))
    hs_two = HierarchySpec(family, Dict(:pair => [left_hc, right_hc]), τ_hc)
    prob_two = offline_problem([model_two]; ids=[:pair], hierarchies=[hs_two])
    @test length(hs_two.members[:pair]) == 2
    @test count(f -> f isa HierarchyFactor && f.child == prob_two.cooks[1].key_z,
                prob_two.graph.factors) == 2
end
println("test_offline.jl: done")
