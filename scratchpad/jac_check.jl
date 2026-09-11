# The smoother (smoother.jl): the stacked layout, the random-walk prior, the segmented physics against the
# 1-D physics, and the fit with no moving site against the batch fit.
#   julia --project=estimator estimator/test_smoother.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3), LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
sites = vcat(prior_phys, [e.site for e in extra]); allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(sites) - 1, length(sites)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
keep6 = t .<= 6 * 3600 + 1e-9
t6, u6 = t[keep6], u[keep6]
d6 = estimator_setup(t6, u6, pit, cfg, prior_phys, names)
iu = findfirst(==("u_leak"), allnames); ik = findfirst(==("k_e"), allnames); ir = findfirst(==("r_max"), allnames)
z0 = vcat(d6.z0, [log(cfg.A_meat), log(0.8 * L_med)])
i2, i4 = findlast(t6 .<= 2 * 3600), findlast(t6 .<= 4 * 3600)

using Printf
ph1 = Physics1D(area_latent=AREA, events=events)
S1, J1, lay = physics_outputs(z0, d6, ph1)
ph0 = PhysicsDrift(AREA, events, DriftLayout(17, []))
S0, J0, _ = physics_outputs(z0, d6, ph0)
nt = lay.nt; nx = lay.nx
blocks = [("pit", lay.oTpit, nt), ("surf", lay.oTsurf, nt), ("T", lay.oT, nx * nt), ("W", lay.oW, nx * nt), ("F", lay.oF, nx * nt), ("pv", lay.opv, nt), ("mass", lay.oMass, nt)]
for (nm, o, len) in blocks
    r = (o + 1):(o + len)
    dS = maximum(abs.(S0[r] .- S1[r]))
    dJ = abs.(J0[r, :] .- J1[r, :]); sc = maximum(abs.(J1[r, :]); dims=1)
    worst = argmax(dJ ./ (abs.(J1[r, :]) .+ 0.01))
    @printf("%-5s max|dS| %.4f   max|dJ| %.4g at column %s (J1 there %.4g, J0 %.4g); column scales max %.3g\n", nm, dS, maximum(dJ), allnames[worst[2]], J1[r, :][worst], J0[r, :][worst], maximum(sc))
end
