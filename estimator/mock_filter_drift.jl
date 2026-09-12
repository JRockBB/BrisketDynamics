# The sequential filter on the drifting mock: one belief over the state and the latents, updated as the
# real-cook instruments' samples arrive, against the plant's own temperatures and its latents in force at each
# hour. Writes data/mock/filter_drift.json for plot_filter_belief.py.
#
# Run:  julia --project=estimator estimator/mock_filter_drift.jl [t_end_h] [step_min] [prior.json]
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
t_end_h = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 16.0
step_min = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 10.0
prior_file = length(ARGS) >= 3 ? ARGS[3] : "prior_next_cook.json"
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_drift.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
L_med = exp(prior_phys[4].p1)
sites = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), PriorSite("lognormal", log(0.8 * L_med), 0.5)])
allnames = vcat(names, ["A_meat", "core_depth"]); idx = Dict(n => i for (i, n) in enumerate(allnames))
AREA, DEPTH = 16, 17
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]),
          dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
nc = RealCookNoise()
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
noise = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0),
             3 => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), 4 => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
q = zeros(length(sites)); for (nm, sd) in DRIFT_SD_PER_HOUR; q[idx[nm]] = sd^2; end
jumps = [(idx[nm], events.t_add, sd^2) for (nm, sd) in JUMP_LOG_SD]
println("the sequential filter on the drifting mock (seed ", m["drift"]["seed"], ") to ", t_end_h, " h, a step of ", step_min, " min; prior ", prior_file)
flush(stdout)
t0 = time()
recs = sequential_filter(readings, masks, t, u, pit, cfg, sites, names, events; step=step_min * 60, noise, cap=nc.hum_cap,
                         depth_latent=DEPTH, area_latent=AREA, q, jumps, verbose=true, t_end=t_end_h * 3600)
@printf("%d steps in %.0f s\n", length(recs), time() - t0)
# the plant's own temperatures (the mock's clean traces) and its latents in force at each step
clean = m["clean"]
x_by_hour = m["truth_by_hour"]["x"]; hours = Float64.(m["truth_by_hour"]["hours"])
tz(x, p) = isbeta(p) ? log(x / (1 - x)) : log(x)
x_true = Float64.(m["truth"]["x"])
z_extra = [log(x_true[16]), log(x_true[17])]
function z_true_at(te)
    k = max(1, findlast(hours .<= te / 3600 - 1e-9))          # the block in force just before te
    xx = Float64.(x_by_hour[k])
    vcat([tz(xx[i], prior_phys[i]) for i in 1:N_PHYS], z_extra)
end
# the plant's surface: the field's surface node from a re-solve of the plant (the mock keeps the probe traces only)
prior_phys_sites = prior_phys
z0_plant = [tz(x_true[i], prior_phys[i]) for i in 1:N_PHYS]
dd = DriftDraw(MersenneTwister(Int(m["drift"]["seed"])), idx)
o = solve_drifting(z0_plant, dd, prior_phys, u, t; pit, cfg, A_meat=x_true[16], events...)
th = [r.t / 3600 for r in recs]
errs = [(r.z .- z_true_at(r.t)) ./ r.z_sd for r in recs]
last_h = [i for i in eachindex(recs) if abs(recs[i].t / 3600 - round(recs[i].t / 3600)) < 1e-6]
for i in last_h
    e = errs[i]
    @printf("  %2.0f h: |err|/sd median %.2f max %.2f (%s); u_leak %+.2f k_e %+.2f h_c %+.2f r_max %+.2f dH_eff %+.2f C_th %+.2f a0 %+.2f\n", recs[i].t / 3600,
            sort(abs.(e))[div(end + 1, 2)], maximum(abs.(e)), allnames[argmax(abs.(e))], e[idx["u_leak"]], e[idx["k_e"]], e[idx["h_c"]], e[idx["r_max"]], e[idx["dH_eff"]], e[idx["C_th"]], e[idx["a0"]])
end
kk(te) = searchsortedlast(t, te + 1e-9)
dst = joinpath(REPO, "data", "mock", "filter_drift.json")
open(dst, "w") do io
    JSON.print(io, Dict("t_h" => th, "step_min" => step_min, "names" => allnames,
                        "pit" => Dict("mean" => [r.T_pit[1] - 273.15 for r in recs], "sd" => [r.T_pit[2] for r in recs]),
                        "centre" => Dict("mean" => [r.T_centre[1] - 273.15 for r in recs], "sd" => [r.T_centre[2] for r in recs]),
                        "surface" => Dict("mean" => [r.T_surface[1] - 273.15 for r in recs], "sd" => [r.T_surface[2] for r in recs]),
                        "z" => [r.z for r in recs], "z_sd" => [r.z_sd for r in recs], "z_true" => [z_true_at(r.t) for r in recs],
                        "iters" => [r.iters for r in recs], "wall_s" => [r.wall for r in recs],
                        "plant" => Dict("t_h" => t ./ 3600, "pit" => Float64.(clean["pit"]) .- 273.15, "centre" => collect(o.Tcore) .- 273.15,
                                        "surface" => collect(o.Tsurf) .- 273.15, "probe" => Float64.(clean["core"]) .- 273.15),
                        "readings" => Dict("pit" => [masks.pit[k] ? readings.pit[k] - 273.15 : nothing for k in eachindex(t)],
                                           "core" => [masks.core[k] ? readings.core[k] - 273.15 : nothing for k in eachindex(t)]),
                        "events_h" => Dict(String(k) => v / 3600 for (k, v) in pairs(events) if k in (:t_wrap, :t_pull, :t_add, :t_spritz))))
end
println("wrote ", dst)
