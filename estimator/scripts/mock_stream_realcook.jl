# The streaming estimator on the real-cook mock: channels at their own
# rates on a two-minute grid (the pit and core probes every two minutes,
# the humidity sensor every ten with its saturation cap as a censored
# channel, the scale every half hour), the lid-open windows dropped, the
# executed events as inputs of the physics, every channel's noise
# identified by the AR node at its own spacing from priors set for a
# ten-minute grid and converted to the grid step, hourly refits on the
# graph engine with the misfit check per channel.
#
# Run:  julia --project=estimator estimator/scripts/mock_stream_realcook.jl
# Writes data/mock/mock_realcook_stream.json.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
names = String.(gate["names"])
d = estimator_setup(t, u, pit, cfg, prior_phys, names)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3),
         LatentSpec("core_depth_fraction", probe_depth_site())]
prior = vcat(prior_phys, [e.site for e in extra])
allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(prior) - 1, length(prior)
x_true = Float64.(m["truth"]["x"])
z_true = [to_z(x, p) for (x, p) in zip(x_true, prior)]
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]),
          t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
ph = Physics1D(area_latent=AREA, events=events)

# the instruments' noise as the estimator starts it: the calibration's pit and core priors
# (a ten-minute grid) converted to this grid's step; assumed values for the humidity and the scale
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
# ρ per step from ρ per calibration step: ρ^(dt/Δ), with the width scaled by the slope of that map
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
priors = Dict("pit" => tostep(cal["pit"], 600.0), "core" => tostep(cal["core"], 600.0),
              "humidity" => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), "mass" => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))   # floors from the instruments' spec
nc = m["noise"]
truth_noise = Dict("pit" => (nc["pit_sd"], exp(-dt / nc["pit_tau"])), "core" => (nc["core_sd"], exp(-dt / nc["core_tau"])),
                   "humidity" => (nc["hum_sd"], exp(-dt / nc["hum_tau"])), "mass" => (nc["mass_sd"], exp(-dt / nc["mass_tau"])))
rd = m["readings"]; mk = m["masks"]
start(np) = (np.σ_mean, np.ρ_mean)
chs = [PitProbe(start(priors["pit"])..., Bool.(mk["pit"]); data=Float64.(rd["pit"])),
       MeatProbe(start(priors["core"])..., Bool.(mk["core"]); depth_latent=DEPTH, data=Float64.(rd["core"])),
       CensoredHumidity(start(priors["humidity"])..., Bool.(mk["humidity"]); cap=Float64(nc["hum_cap"]), data=Float64.(rd["humidity"])),
       MassScale(start(priors["mass"])..., Bool.(mk["mass"]); data=Float64.(rd["mass"]))]
np = Dict(1 => priors["pit"], 2 => priors["core"], 3 => priors["humidity"], 4 => priors["mass"])
chnames = ["pit", "core", "humidity", "mass"]
edges = 3600.0 .* collect(2:2:floor(Int, events.t_pull / 3600))        # two-hourly to the pull

println("plant: ", m["plant"], "; grid step ", dt, " s; channel spacings (steps): ",
        [E.channel_gap(ch.mask) for ch in chs], "; true noise per step ", truth_noise)
flush(stdout)
t0 = time()
recs = stream_fit(chs, prior, d, edges; physics=ph, noise=np, noise_rounds=1, levelled=true, verbose=true)   # the noise carries across batches: one round each; the level as a process
batches = [Dict("t_h" => r.t / 3600, "z" => r.z, "sd" => r.sd, "err_sd" => (r.z .- z_true) ./ r.sd, "F" => r.F,
                "iters" => r.iters, "wall_s" => r.wall, "misfit" => r.misfit,
                "noise" => Dict(chnames[i] => Dict(pairs(v)) for (i, v) in r.noise),
                "level" => Dict(chnames[i] => (ch isa Levelled ? ch.level : nothing) for (i, ch) in enumerate(r.channels))) for r in recs]
last = recs[end]
e = abs.((last.z .- z_true) ./ last.sd)
mt = last.misfit.total
@printf("final |err|/sd: max %.2f (%s), median %.2f; residual power/row %.3f, z_chi2 %.2f, z_rho1 %.2f; wall %.0f s\n",
        maximum(e), allnames[argmax(e)], sort(e)[div(end + 1, 2)], mt.ss / mt.n, mt.z_chi2, mt.z_rho1, time() - t0)
for c in last.misfit.channels
    @printf("  %-20s n %4d  z_chi2 %6.2f  z_rho1 %s\n", c.name, c.n, c.z_chi2, c.z_rho1 === nothing ? "-" : @sprintf("%6.2f", c.z_rho1))
end
for (i, v) in sort(collect(last.noise); by=first)
    tr = truth_noise[chnames[i]]
    @printf("  %-8s σ %.4g ± %.2g (truth %.4g, prior %.4g)   ρ/step %.3f ± %.3f (truth %.3f, prior %.3f)\n",
            chnames[i], v.σ, v.σ_sd, tr[1], np[i].σ_mean, v.ρ, v.ρ_sd, tr[2], np[i].ρ_mean)
end
flush(stdout)
dst = joinpath(REPO, "data", "mock", "mock_realcook_stream.json")
open(dst, "w") do io
    JSON.print(io, Dict("plant" => m["plant"], "names" => allnames, "z_true" => z_true, "x_true" => x_true,
                        "sets" => [Dict("label" => "pit + core + humidity (censored) + scale, at their rates", "batches" => batches, "channels" => chnames)],
                        "noise_truth" => Dict(k => Dict("sigma" => v[1], "rho" => v[2]) for (k, v) in truth_noise),
                        "noise_prior" => Dict(k => Dict("sigma" => v.σ_mean, "rho" => v.ρ_mean) for (k, v) in priors),
                        "edges_h" => edges ./ 3600))
end
println("wrote ", dst)
