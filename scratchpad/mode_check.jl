# Is the estimator's overconfidence a wrong mode, or the model's? On the last closed-loop run's readings
# (regenerated from the executed schedule) at the end of the cook: the fit objective at the truth, at the run's
# final estimate, and a refit started at the truth, with the run's noise model.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf, Random, Statistics
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json")); gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
rec = JSON.parsefile(joinpath(REPO, "data", "mock", "closed_loop_realcook.json"))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
names = String.(gate["names"])
prior = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), probe_depth_site()])
AREA, DEPTH = 16, 17
GRID = collect(0.0:120.0:30 * 3600.0); nc = RealCookNoise()
x_true = Float64.(m["truth"]["x"]); θ_true, φ_true, A_true, depth_true = x_true[1:8], x_true[9:15], x_true[16], x_true[17]; L_true = θ_true[4]
z_true = Float64.(rec["z_true"]); z_hat = Float64.(rec["checks"][end]["posterior_z"]); sd_hat = Float64.(rec["checks"][end]["posterior_sd"])
ex = rec["executed"]; t10 = Float64.(ex["t_h"]) .* 3600
u10 = [v === nothing ? NaN : Float64(v) for v in ex["u"]]; u10 = [isnan(v) ? u10[findlast(!isnan, u10)] : v for v in u10]
u = [ti >= t10[end] ? u10[end] : (k = searchsortedlast(t10, ti); k >= length(t10) ? u10[end] : u10[k] + (u10[k + 1] - u10[k]) * (ti - t10[k]) / (t10[k + 1] - t10[k])) for ti in GRID]
sp = ex["spritz"]
ev = (t_wrap=Float64(ex["t_wrap_h"]) * 3600, t_pull=Float64(ex["t_pull_h"]) * 3600, t_add=ex["t_add_h"] === nothing ? BrisketDynamics.ADD_NEVER : Float64(ex["t_add_h"]) * 3600,
      t_spritz=sp === nothing ? BrisketDynamics.SPRITZ_NEVER : Float64(sp["t_h"]) * 3600, dt_spritz=sp === nothing ? 3600.0 : Float64(sp["dt_h"]) * 3600, m_spritz=sp === nothing ? 0.05 : Float64(sp["m"]))
o = monolith_solve(θ_true, φ_true, u, GRID, GRID; pit, cfg, A_meat=A_true, ev...)
r, _, masks, _ = realcook_readings(o, GRID, ev, nc, MersenneTwister(23); A=A_true, L=L_true, depth=probe_depth(L_true, depth_true), cfg)
t_end = Float64(ex["t_serve_h"]) * 3600 - 3600            # the last check
keep = GRID .<= t_end + 1e-9
d = estimator_setup(GRID, u, pit, cfg, prior_phys, names)
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json")); dt = 120.0
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
np = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0), 3 => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3), 4 => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3))
last_noise = rec["checks"][end]["noise"]
nz(k) = (last_noise[k]["sigma"], last_noise[k]["rho"])
chs = [PitProbe(nz("pit")..., masks.pit .& keep; data=r.pit), MeatProbe(nz("core")..., masks.core .& keep; depth_latent=DEPTH, data=r.core),
       CensoredHumidity(nz("humidity")..., masks.humidity .& keep; cap=nc.hum_cap, data=r.humidity), MassScale(nz("mass")..., masks.mass .& keep; data=r.mass)]
ph = Physics1D(area_latent=AREA, events=ev)
println("readings to ", t_end / 3600, " h; the run's final noise: ", Dict(k => round.(nz(k); digits=3) for k in ("pit", "core", "humidity", "mass")))
zsd(z) = (z .- z_true) ./ sd_hat
for (lab, c0) in (("the run's final estimate", z_hat), ("the truth", z_true))
    res = solve_map_channels(chs, prior, d; c0, physics=ph, engine=:graph, noise=nothing, verbose=false, iters=40)
    e = abs.(res.z .- z_true) ./ sqrt.(diag(res.cov))
    @printf("refit from %-26s F %9.3f  converged %s  |err|/sd median %.2f max %.2f (%s); misfit z %.2f\n", lab, res.F, res.converged, sort(e)[div(end + 1, 2)], maximum(e), vcat(names, ["A_meat", "depth"])[argmax(e)], res.misfit.total.z_chi2)
    @printf("   moved from its start by (in the run's sd): median %.2f max %.2f\n", median(abs.((res.z .- c0) ./ sd_hat)), maximum(abs.((res.z .- c0) ./ sd_hat)))
end

# the same fit with the plant's true noise per channel: is the overconfidence the noise model's?
truth_noise = Dict("pit" => (nc.pit_sd, exp(-dt / nc.pit_tau)), "core" => (nc.core_sd, exp(-dt / nc.core_tau)),
                   "humidity" => (nc.hum_sd, exp(-dt / nc.hum_tau)), "mass" => (nc.mass_sd, exp(-dt / nc.mass_tau)))
chs_true = [PitProbe(truth_noise["pit"]..., masks.pit .& keep; data=r.pit), MeatProbe(truth_noise["core"]..., masks.core .& keep; depth_latent=DEPTH, data=r.core),
            CensoredHumidity(truth_noise["humidity"]..., masks.humidity .& keep; cap=nc.hum_cap, data=r.humidity), MassScale(truth_noise["mass"]..., masks.mass .& keep; data=r.mass)]
res = solve_map_channels(chs_true, prior, d; c0=z_hat, physics=ph, engine=:graph, noise=nothing, verbose=false, iters=40)
sd_true = sqrt.(diag(res.cov)); e = abs.(res.z .- z_true) ./ sd_true
@printf("refit with the TRUE noise:            F %9.3f  |err|/sd median %.2f max %.2f (%s); misfit z %.2f\n", res.F, sort(e)[div(end + 1, 2)], maximum(e), vcat(names, ["A_meat", "depth"])[argmax(e)], res.misfit.total.z_chi2)
allnames = vcat(names, ["A_meat", "depth"])
println("   posterior sd, true noise / the run's, per latent: ", Dict(n => round(a / b; digits=2) for (n, a, b) in zip(allnames, sd_true, sd_hat)))
println("   |err| in the run's sd vs in the true-noise sd: ", Dict(n => (round(abs(res.z[i] - z_true[i]) / sd_hat[i]; digits=1), round(e[i]; digits=1)) for (i, n) in enumerate(allnames) if e[i] > 1.5 || abs(res.z[i] - z_true[i]) / sd_hat[i] > 1.5))
