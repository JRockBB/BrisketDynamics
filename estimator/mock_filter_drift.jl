# The filter on the graph on the drifting mock: one belief over the state, the latents and the instruments'
# discrepancies, updated as the real-cook instruments' samples arrive, against the plant's own temperatures and
# its latents in force at each hour. Writes data/mock/filter_drift.json for plot_filter_belief.py.
#
# Run:  julia --project=estimator estimator/mock_filter_drift.jl [t_end_h] [prior.json] [fault: core probe stuck from hour h, 0 = none]
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
prior_file = length(ARGS) >= 2 ? ARGS[2] : "prior_next_cook.json"
fault_h = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.0
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_drift.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", prior_file), cfg)
sites, allnames, idx, prior_phys = blk.sites, blk.names, blk.idx, blk.prior_phys
AREA, DEPTH = blk.area, blk.depth
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]),
          dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
nc = RealCookNoise()
np0 = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
noise = [np0[i] for i in 1:4]
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
if fault_h > 0                                                    # the core probe sticks at its last reading from fault_h on
    kf = findfirst(k -> t[k] >= fault_h * 3600 && masks.core[k], eachindex(t))
    readings.core[kf:end] .= readings.core[kf]
end
chs = realcook_channels(readings, masks, np0; cap=nc.hum_cap, depth_latent=DEPTH)
dr = drift_rates(blk.file)
q = zeros(length(sites)); for (nm, sd) in dr.sd_per_hour; q[idx[nm]] = sd^2; end
jumps = [(idx[nm], events.t_add, sd^2) for (nm, sd) in dr.jump_log_sd]
println("the filter on the drifting mock (seed ", m["drift"]["seed"], ") to ", t_end_h, " h, updates on arrival; prior ", prior_file,
        fault_h > 0 ? "; the core probe stuck from $(fault_h) h" : "")
flush(stdout)
t0 = time()
recs = sequential_filter(chs, t, u, pit, cfg, sites, allnames, events; noise, area_latent=AREA, q, jumps, diagnostics=true, verbose=true, t_end=t_end_h * 3600)
@printf("%d updates in %.0f s\n", length(recs), time() - t0)
# the plant's own temperatures (the mock's clean traces) and its latents in force at each step
clean = m["clean"]
x_by_hour = m["truth_by_hour"]["x"]; hours = Float64.(m["truth_by_hour"]["hours"])
x_true = Float64.(m["truth"]["x"])
z_extra = [log(x_true[16]), log(x_true[17])]
function z_true_at(te)
    k = max(1, findlast(hours .<= te / 3600 - 1e-9))          # the block in force just before te
    xx = Float64.(x_by_hour[k])
    vcat(to_z(xx[1:N_PHYS], prior_phys), z_extra)
end
# the plant's surface: the field's surface node from a re-solve of the plant (the mock keeps the probe traces only)
z0_plant = to_z(x_true[1:N_PHYS], prior_phys)
dd = DriftDraw(MersenneTwister(Int(m["drift"]["seed"])), idx)
o = solve_drifting(z0_plant, dd, prior_phys, u, t; pit, cfg, A_meat=x_true[16], events...)
th = [r.t / 3600 for r in recs]
errs = [(r.z .- z_true_at(r.t)) ./ r.z_sd for r in recs]
for i in eachindex(recs)
    (i == length(recs) || floor(recs[i + 1].t / 3600 + 1e-9) > floor(recs[i].t / 3600 + 1e-9)) || continue    # the last update of each hour
    e = errs[i]
    @printf("  %2.0f h: |err|/sd median %.2f max %.2f (%s); u_leak %+.2f k_e %+.2f h_c %+.2f r_max %+.2f dH_eff %+.2f C_th %+.2f a0 %+.2f; levels %s%s\n", floor(recs[i].t / 3600 + 1e-9),
            sort(abs.(e))[div(end + 1, 2)], maximum(abs.(e)), allnames[argmax(abs.(e))], e[idx["u_leak"]], e[idx["k_e"]], e[idx["h_c"]], e[idx["r_max"]], e[idx["dH_eff"]], e[idx["C_th"]], e[idx["a0"]],
            join([@sprintf("%.1f", l) for l in recs[i].level], " "), any(recs[i].faulty) ? " FAULT " * join(["pit", "core", "humidity", "scale"][findall(recs[i].faulty)], ",") : "")
end
kk(te) = searchsortedlast(t, te + 1e-9)
dst = joinpath(REPO, "data", "mock", fault_h > 0 ? "filter_drift_fault.json" : "filter_drift.json")
open(dst, "w") do io
    JSON.print(io, Dict("t_h" => th, "names" => allnames, "fault_h" => fault_h,
                        "pit" => Dict("mean" => [r.T_pit[1] - 273.15 for r in recs], "sd" => [r.T_pit[2] for r in recs]),
                        "centre" => Dict("mean" => [r.T_centre[1] - 273.15 for r in recs], "sd" => [r.T_centre[2] for r in recs]),
                        "surface" => Dict("mean" => [r.T_surface[1] - 273.15 for r in recs], "sd" => [r.T_surface[2] for r in recs]),
                        "z" => [r.z for r in recs], "z_sd" => [r.z_sd for r in recs], "z_true" => [z_true_at(r.t) for r in recs],
                        "disc" => [r.disc for r in recs], "disc_sd" => [r.disc_sd for r in recs],
                        "level" => [r.level for r in recs], "faulty" => [r.faulty for r in recs], "noise" => [[[v.σ, v.ρ] for v in r.noise] for r in recs],
                        "nis" => [r.nis for r in recs], "dof" => [r.dof for r in recs],
                        "iters" => [r.iters for r in recs], "wall_s" => [r.wall for r in recs],
                        "plant" => Dict("t_h" => t ./ 3600, "pit" => Float64.(clean["pit"]) .- 273.15, "centre" => collect(o.Tcore) .- 273.15,
                                        "surface" => collect(o.Tsurf) .- 273.15, "probe" => Float64.(clean["core"]) .- 273.15),
                        "readings" => Dict("pit" => [masks.pit[k] ? readings.pit[k] - 273.15 : nothing for k in eachindex(t)],
                                           "core" => [masks.core[k] ? readings.core[k] - 273.15 : nothing for k in eachindex(t)]),
                        "events_h" => Dict(String(k) => v / 3600 for (k, v) in pairs(events) if k in (:t_wrap, :t_pull, :t_add, :t_spritz))))
end
println("wrote ", dst)
