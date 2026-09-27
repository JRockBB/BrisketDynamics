using Pkg
Pkg.activate(joinpath("C:/Users/jmurp/.julia/dev/BrisketDynamics/estimator/scripts", ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath("C:/Users/jmurp/.julia/dev/BrisketDynamics/estimator/scripts", "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath("C:/Users/jmurp/.julia/dev/BrisketDynamics/estimator/scripts", "..", ".."))
prior_file = length(ARGS) >= 1 ? ARGS[1] : "prior_next_cook.json"
n_cooks = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
seed0 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
out_name = length(ARGS) >= 4 ? ARGS[4] : "coverage_system_$(seed0).json"
edges_to_h = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 14.0
drift_on = length(ARGS) >= 6 ? ARGS[6] == "1" : false
cal_file = length(ARGS) >= 7 ? ARGS[7] : "default"
LAG = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 3
stale_ok = length(ARGS) >= 9 ? ARGS[9] == "1" : false
als_lag_h = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 2.0
noise_mode = length(ARGS) >= 11 ? ARGS[11] : "matched"
noise_mode in ("matched", "prior") || error("noise must be matched or prior, not $noise_mode")

m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", prior_file), cfg)
PIECE = :brisket
sys = CookSystem(Slab1D(PIECE; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[PIECE])
blk.names == blk_old.names || error("the system's latent block is not the filter's ($(blk.names) against $(blk_old.names))")
sites, allnames = blk.sites, blk.names
idx = Dict(nm => i for (i, nm) in enumerate(allnames))
AREA, DEPTH = latent_index(blk, "A_meat"), latent_index(blk, "core_depth_fraction")
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
cook_events = [CookEvent(:wrap, PIECE, events.t_wrap), CookEvent(:pull, PIECE, events.t_pull), CookEvent(:add, :pit, events.t_add),
               CookEvent(:spritz, PIECE, events.t_spritz; dt=events.dt_spritz, m=events.m_spritz)]
edges = unique(vcat(3600.0 .* collect(2:2:floor(Int, edges_to_h)), [3600.0 * edges_to_h]))     # every two hours, and the run's end
t_end = edges[end]
nc = RealCookNoise()
np0 = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
dr = drift_rates(blk_old.file; sites, idx)                                                  # the plant's rates (drift.jl's, in the sites' coordinates)
# the drift the smoother is told when the plant drifts: the walking sites at the plant's rates, the jumping sites at the add
# (a copy of the moving latents per window node tied by the random walk, online.jl); none on the static plant
moving_names = drift_on ? sort(unique(vcat(collect(keys(dr.sd_per_hour)), collect(keys(dr.jump_log_sd))))) : String[]
drift_spec = drift_on ? (moving=[idx[nm] for nm in moving_names], sd_per_hour=[get(dr.sd_per_hour, nm, 0.0) for nm in moving_names],
                         jumps=[(idx[nm], events.t_add, sd) for (nm, sd) in dr.jump_log_sd]) : nothing

# the calibration the smoother runs with: the state's rates and, when the file carries them, the sensors' noise
sn, cal_sensors = cal_file == "default" ? (StateNoise(), nothing) :
                  load_system_calibration(isabspath(cal_file) ? cal_file : joinpath(REPO, "data", "calibration", cal_file); strict=!stale_ok)
tau_of(npr) = -dt / log(clamp(npr.ρ_mean, 1e-6, 0.999))
base_noise = noise_mode == "matched" ?
    Dict(:pit => (white_sd=0.1, ar_sd=nc.pit_sd, tau=nc.pit_tau), :core => (white_sd=0.1, ar_sd=nc.core_sd, tau=nc.core_tau),
         :humidity => (white_sd=10.0, ar_sd=nc.hum_sd, tau=nc.hum_tau), :mass => (white_sd=0.005, ar_sd=nc.mass_sd, tau=nc.mass_tau)) :
    Dict(:pit => (white_sd=0.1, ar_sd=np0[1].σ_mean, tau=tau_of(np0[1])), :core => (white_sd=0.1, ar_sd=np0[2].σ_mean, tau=tau_of(np0[2])),
         :humidity => (white_sd=10.0, ar_sd=np0[3].σ_mean, tau=tau_of(np0[3])), :mass => (white_sd=0.005, ar_sd=np0[4].σ_mean, tau=tau_of(np0[4])))
sensor_noise = cal_sensors === nothing || isempty(cal_sensors) ? base_noise : Dict(id => get(cal_sensors, id, v) for (id, v) in base_noise)
SENSOR_IDS = (:pit, :core, :humidity, :mass)                                                # the record's channels 1 to 4, the filter's order
observable_of = Dict(:pit => PitTemperature(), :core => TemperatureAt(PIECE; depth_latent=:core_depth_fraction), :humidity => ChamberVapor(), :mass => TotalMass())
sensors = [SensorSpec(id, observable_of[id], NoiseModel(; white_sd=sensor_noise[id].white_sd, ar_sd=sensor_noise[id].ar_sd, tau=sensor_noise[id].tau,
                                                      cap=(id === :humidity ? nc.hum_cap : Inf))) for id in SENSOR_IDS]
fingerprint = backend_fingerprint()
bl = belief_layout(sys)
group_names = [E._als_group_name(bl, g) for (g, _) in bl.groups]                            # the filter's twelve names
group_ranges = [rg for (_, rg) in bl.groups]
# a slab field's noise is the bulk and FIELD_MODES smooth modes (system_state.jl): its NEES is scored on those coefficients
field_basis = Dict(g => E.field_mode_basis(bl, gfull) for (g, (gfull, _)) in zip(group_names, bl.groups))
group_dof = Dict{String,Int}(g => (sn.fields === :offset && length(rg) > 1 && field_basis[g] !== nothing ? size(field_basis[g], 2) : length(rg)) for (g, rg) in zip(group_names, group_ranges))
ip = first(E.belief_index(bl, :pit, :T_pit)); rT = E.belief_index(bl, PIECE, :T); ic, is = first(rT), last(rT)
ia0 = idx["a0"]; keep16 = [i for i in 1:length(sites) if i != ia0]
p_pack, play_pack = pack_params(sys, [zeros(8)], [cfg.A_meat], vcat(zeros(6), 0.5), u[1:2]; events=cook_events)
PIT_ICS_FIELDS = (:Tpit0, :mfuel0, :mpan0, :Tpan0, :Tfb0)

cooks = Any[]; skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    write(io, json_document(Dict("plant" => "monolith (1-D) in segments, parameters " * (drift_on ? "drifting (drift.jl)" : "constant (the drift draw zeroed)") * "; real-cook instruments",
                        "backend" => "system", "estimator" => "the square-root fixed-lag smoother over a CookSystem of one slab (online.jl), lag $LAG",
                        "names" => allnames, "prior_file" => prior_file, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped,
                        "drift" => Dict("on" => drift_on, "sd_per_hour" => dr.sd_per_hour, "jump_log_sd" => dr.jump_log_sd,
                                        "stated" => "in the logit of a fraction or the log, carried into each site's coordinate by drift_scale"),
                        "q" => [drift_on ? get(dr.sd_per_hour, nm, 0.0)^2 : 0.0 for nm in allnames],
                        "latents_in_window" => drift_on ? "the moving latents ($(join(moving_names, ", "))) a copy per window node tied by the random walk at the plant's rates, the jumps at the add; the rest static" : "static",
                        "state_noise" => state_noise_record(bl, sn), "state_noise_file" => cal_file,
                        "sensors" => Dict(string(id) => Dict("white_sd" => v.white_sd, "ar_sd" => v.ar_sd, "tau" => v.tau) for (id, v) in sensor_noise), "noise_mode" => noise_mode,
                        "physics" => fingerprint, "physics_only" => physics_fingerprint(), "adapt_level" => false, "groups" => group_names, "group_dof" => group_dof,
                        "n_sigma" => 0, "lag" => LAG, "stale_ok" => stale_ok, "als_lag_h" => als_lag_h)))
end
println("the factor-graph backend: prior ", prior_file, "; ", n_cooks, " cooks from seed ", seed0, "; edges to ", edges_to_h, " h; the plant ",
        drift_on ? "drifting" : "static", "; lag ", LAG, "; the state noise from ", cal_file, "; the sensor noise ", noise_mode,
        cal_sensors === nothing || isempty(cal_sensors) ? "" : " overridden by the calibration file", "; backend ", fingerprint,
        stale_ok ? "; stale calibration allowed (the tune round)" : "", als_lag_h > 0 ? "; the innovation autocovariances accumulated to $(als_lag_h) h" : "", "; record ", out_name)
println("  rates [units/sqrt(h)]: ", join(["$g $(getfield(sn, Symbol(g)))" for g in STATE_GROUPS], ", "))
println("  initial sds: ", join(["$g $(getfield(sn, Symbol(g * "0")))" for g in STATE_GROUPS], ", "))
println("  sensors: ", join(["$id white $(sensor_noise[id].white_sd) ou $(sensor_noise[id].ar_sd) tau $(round(Int, sensor_noise[id].tau)) s" for id in SENSOR_IDS], "; "))
flush(stdout)

"The plant's state at grid point `k` of the segments output as the belief's physical state and coordinates."
function truth_belief(o, k)
    ics, a = handoff(o, k)
    pf = Tuple(f for f in keys(ics) if !(f in PIT_ICS_FIELDS))                             # the piece's restart fields (the latch is not carried: the wrap is a time)
    piece_ics = NamedTuple{pf}(Tuple(getfield(ics, f) for f in pf))
    pit_ics = (Tpit0=ics.Tpit0, mfuel0=ics.mfuel0, mpan0=ics.mpan0, Tpan0=ics.Tpan0, Tfb0=ics.Tfb0, ignited=clamp(a, 1e-6, 1 - 1e-6))
    x_sys = initial_state(sys, p_pack, play_pack; ics=[piece_ics], pit_ics)
    xb = belief_from_system(x_sys, bl)
    return xb, belief_encode(xb, bl)
end
function nees(e, C)
    F = cholesky(Matrix(Symmetric(C)); check=false)
    issuccess(F) ? dot(e, F \ e) : NaN
end
function nees_corr(e, C)                                   # in the block's correlation form (a field block's entries span orders of magnitude)
    s = sqrt.(max.(diag(C), 1e-300))
    F = cholesky(Matrix(Symmetric(C ./ (s * s'))); check=false)
    issuccess(F) ? dot(e ./ s, F \ (e ./ s)) : NaN
end
# a field under :offset carries the bulk and the smooth modes' draws: scored on those coefficients (a = Bᵀe/n, the
# columns orthogonal with squared norm n) against the plant's, one degree of freedom each; otherwise node by node
function group_nees(ey, Cy)
    Dict(g => begin
             B = field_basis[g]
             if B !== nothing && group_dof[g] < length(rg)
                 A = B' ./ length(rg)
                 ea = A * ey[rg]; Ca = A * Cy[rg, rg] * A'
                 nees(ea, Ca) / size(B, 2)
             else
                 nees_corr(ey[rg], Cy[rg, rg]) / length(rg)
             end
         end for (g, rg) in zip(group_names, group_ranges))
end

