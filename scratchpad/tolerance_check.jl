# the model against the plant at the SAME parameters: the estimator's physics_stack (its tolerances) versus the
# mock's monolith_solve (its defaults), in the readings' units, against each channel's noise
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
d = estimator_setup(t, u, pit, cfg, prior_phys, String.(gate["names"]))
println("estimator tolerances: abstol ", d.abstol, " reltol ", d.reltol)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
x = Float64.(m["truth"]["x"]); θ, φ, A, L = x[1:8], x[9:15], x[16], x[4]
o_plant = monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, events...)                                        # the mock's call
o_model = monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, abstol=d.abstol, reltol=d.reltol, events...)     # the estimator's call
nx = size(o_plant.W_field, 1); w = E.fv_weights(nx)
mass(o) = [E.RHO_DRY_EST * A * L * (1 + dot(w, view(o.W_field, :, k)) + dot(w, view(o.F_field, :, k))) for k in axes(o.W_field, 2)]
keep = t .<= 14 * 3600
dm = (mass(o_model) .- mass(o_plant))[keep]; dc = (o_model.Tcore .- o_plant.Tcore)[keep]; dp = (o_model.Tpit .- o_plant.Tpit)[keep]; dh = (o_model.pv_pit .- o_plant.pv_pit)[keep]
@printf("model - plant to 14 h:  mass max %.4f kg mean %+.4f (noise 0.020)   core max %.3f C mean %+.3f (noise 1.2)   pit max %.3f mean %+.3f (noise 1.5)   humidity max %.1f Pa mean %+.1f (noise 100)\n",
        maximum(abs.(dm)), sum(dm) / length(dm), maximum(abs.(dc)), sum(dc) / length(dc), maximum(abs.(dp)), sum(dp) / length(dp), maximum(abs.(dh)), sum(dh) / length(dh))
# and the same at a tighter tolerance on both sides: is either converged?
o_tight = monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, abstol=1e-9, reltol=1e-9, events...)
dmt = (mass(o_plant) .- mass(o_tight))[keep]; dmm = (mass(o_model) .- mass(o_tight))[keep]
@printf("against a 1e-9 solve: the plant's mass off by max %.4f kg mean %+.4f; the model's by max %.4f mean %+.4f\n", maximum(abs.(dmt)), sum(dmt) / length(dmt), maximum(abs.(dmm)), sum(dmm) / length(dmm))
