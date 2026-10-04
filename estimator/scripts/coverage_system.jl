# The coverage and consistency test of the FACTOR-GRAPH backend (SystemModel, online.jl: the square-root fixed-lag
# smoother over a CookSystem of one slab) on the RxInfer bench's plants: the same cooks as coverage_filter.jl
# (truths drawn from the pilot's prior per seed, the monolith solved in segments, read by the real-cook instruments
# with their own noise), the plant STATIC by default (the drift draw zeroed: the same seeds, the same truths; `drift 1`
# walks it as the RxInfer bench does), the smoother run arrival by arrival to the last edge at the pilot's lag, its
# belief of the latents scored against the truth in force at every batch edge, its belief of the pit, centre and
# surface temperatures and of every state group against the plant's at every arrival, and the consistency tests at
# every arrival: the normalized innovation squared against the predicted innovation covariance with its degrees of
# freedom, the normalized estimation error squared of the latents and of the three temperatures, the standardized
# innovations per sensor. The record's schema is coverage_filter.jl's, so plot_coverage.py (key err_sd_graphfilter)
# and plot_filter_consistency.py render it unchanged; the bench driver (bench_system.jl) pools it.
#
# Run:  julia --project=estimator estimator/scripts/coverage_system.jl [prior.json] [n_cooks] [seed0] [out.json] [edges_to_h] [drift 0/1]
#                                                                       [calibration.json|default] [lag] [stale_ok 0/1] [als_lag_h] [noise matched|prior]
#   drift: 0 (the default) holds the plant's latents constant; 1 walks them as drift.jl draws, and the smoother carries
#          the walking and jumping sites as a copy per window node tied by the random walk at the plant's rates (online.jl)
#   calibration.json: data/calibration/state_noise_system.json (or an absolute path): the state's rates and the sensors'
#          noise the smoother runs with; "default" the StateNoise defaults and the noise mode's sensor noise
#   lag: the smoother's window behind the newest node (3, the pilot's, by default)
#   stale_ok: 1 lets a calibration whose backend fingerprint is not the tree's stand (the tune round); else strict
#   als_lag_h: the lag window, in hours, of the innovation-autocovariance accumulator (2 by default; 0 runs without it);
#          its sufficient statistics are written per cook under "als" for bench_system.jl's identification
#   noise: matched (the default): every sensor's OU sd and correlation time are the mock instruments' own (RealCookNoise)
#          with the pilot loop's white floors; prior: the calibration's noise prior (noise_prior_real.json), the RxInfer bench's
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
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
    pf = Tuple(f for f in keys(ics) if !(f in PIT_ICS_FIELDS))                             # the piece's restart fields
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

for c in 1:n_cooks
    seed = seed0 + c - 1
    rng = MersenneTwister(2000 + seed)
    z0 = draw_truth(rng, sites)
    dd0 = DriftDraw(rng, idx; sites)                                                  # drawn on every seed so the instruments' draw is the RxInfer bench's
    dd = drift_on ? dd0 : DriftDraw(Dict(k => zero(v) for (k, v) in dd0.inc), Dict(k => 0.0 for (k, v) in dd0.jump))
    seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
    xs = [from_z(z, sites) for z in blocks]
    A, depth_fraction = xs[1][AREA], xs[1][DEPTH]
    depth = probe_depth(xs[1][4], depth_fraction)
    t0 = time()
    o = try
        solve_segments([x[1:8] for x in xs], [x[9:15] for x in xs], u, t, seg_edges; pit, cfg, A_meat=A, events...)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the plant failed (%s); skipped\n", c, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    all(isfinite, o.Tcore) || (@printf("cook %d (seed %d): the plant is not finite; skipped\n", c, seed); push!(skipped, seed); save(); continue)
    readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=xs[1][4], depth=depth, cfg)
    k_end = searchsortedlast(t, t_end + 1e-9)
    recs = MeasurementRecord[]
    for k in 1:k_end, id in SENSOR_IDS
        getfield(masks, id)[k] && push!(recs, MeasurementRecord(id, t[k], getfield(readings, id)[k]))
    end
    arrivals = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
    clog = CookLog(t, u; events=cook_events, sensors, measurements=recs)
    model = SystemModel(sys, clog, blk; sn)
    feed = als_lag_h > 0 ? ALSFeed(model; lag_s=als_lag_h * 3600, bin_s=dt) : nothing
    sm = online_init(model; lag=LAG, als=feed, drift=drift_spec)
    kk(te) = searchsortedlast(t, te + 1e-9)
    # the whitened residuals at the accepted point (the record's `resid`) and the smoothed past (a node's belief as it left the
    # window, scored against the plant at its own time): the graph's own consistency reads beside NIS and NEES
    resid_dict(rs) = rs === nothing ? nothing :
        Dict("dyn" => rs.dyn, "dyn_window" => rs.dyn_window, "obs" => Dict(string(id) => v for (id, v) in rs.obs), "ou" => Dict(string(id) => v for (id, v) in rs.ou),
             "rows" => rs.rows, "cols" => rs.cols)
    function past_dict(p)
        p === nothing && return nothing
        kp = kk(p.t); _, y_true_p = truth_belief(o, kp)
        eTp = [p.T_pit[1] - o.Tpit[kp], p.pieces[PIECE].T_centre[1] - o.Tcore[kp], p.pieces[PIECE].T_surface[1] - o.Tsurf[kp]]
        ezp = p.z .- truth_at(z0, dd, p.t, events.t_add)
        return Dict("t_h" => p.t / 3600, "k" => kp, "nees_g" => group_nees(p.y .- y_true_p, p.y_cov),
                    "pit" => [p.T_pit[1], p.T_pit[2], o.Tpit[kp]], "centre" => [p.pieces[PIECE].T_centre[1], p.pieces[PIECE].T_centre[2], o.Tcore[kp]],
                    "surface" => [p.pieces[PIECE].T_surface[1], p.pieces[PIECE].T_surface[2], o.Tsurf[kp]],
                    "nees_T" => nees(eTp, p.x_cov[[ip, ic, is], [ip, ic, is]]), "nees_z16" => nees(ezp[keep16], p.z_cov[keep16, keep16]))
    end
    # one arrival's record scored against the plant at its grid point, as it comes (the record itself, with its covariances,
    # is dropped once scored: a cook of four hundred arrivals would otherwise hold them all)
    function score(r)
        k = kk(r.t)
        zt = truth_at(z0, dd, r.t, events.t_add); ez = r.z .- zt
        xb_true, y_true = truth_belief(o, k)
        eT = [r.T_pit[1] - o.Tpit[k], r.pieces[PIECE].T_centre[1] - o.Tcore[k], r.pieces[PIECE].T_surface[1] - o.Tsurf[k]]
        T_cov = r.x_cov[[ip, ic, is], [ip, ic, is]]
        innov = Dict(string(findfirst(==(id), SENSOR_IDS)) => [v] for (id, v) in r.innov_std)
        return Dict("t_h" => r.t / 3600, "k" => k, "updated" => r.committed, "nees_g" => group_nees(r.y .- y_true, r.y_cov),
                    "resid" => resid_dict(r.resid), "past" => past_dict(r.past), "F" => r.F,
                    "pit" => [r.T_pit[1], r.T_pit[2], o.Tpit[k]], "centre" => [r.pieces[PIECE].T_centre[1], r.pieces[PIECE].T_centre[2], o.Tcore[k]],
                    "surface" => [r.pieces[PIECE].T_surface[1], r.pieces[PIECE].T_surface[2], o.Tsurf[k]],
                    "nis" => r.nis, "dof" => r.dof, "innov" => innov,
                    "nees_z" => nees(ez, r.z_cov), "nees_z16" => nees(ez[keep16], r.z_cov[keep16, keep16]), "nees_T" => nees(eT, T_cov),
                    "level" => ones(length(SENSOR_IDS)), "faulty" => falses(length(SENSOR_IDS)),
                    "noise" => [[sensor_noise[id].ar_sd, exp(-dt / sensor_noise[id].tau)] for id in SENSOR_IDS], "wall_s" => r.wall,
                    "iters" => r.iters, "rounds" => r.attempts, "converged" => r.converged, "committed" => r.committed, "gradient" => r.gradient,
                    "n_nodes" => r.n_nodes)
    end
    out = NamedTuple[]                                  # per arrival what the edges and the printout need: the time, the latents' belief, whether it committed
    temps = Any[]
    failed = nothing
    for grp in arrivals
        r = try
            assimilate!(sm, grp)
        catch err
            err isa InterruptException && rethrow()
            failed = err; break
        end
        push!(temps, score(r))
        push!(out, (t=r.t, z=copy(r.z), z_cov=copy(r.z_cov), committed=r.committed))
    end
    if failed !== nothing
        @printf("cook %d (seed %d): the smoother failed at %.2f h (%s); skipped\n", c, seed, isempty(out) ? 0.0 : out[end].t / 3600, sprint(showerror, failed)[1:min(end, 120)])
        push!(skipped, seed); save(); continue
    end
    wall = time() - t0
    at_edge(te) = out[argmin(abs.([r.t for r in out] .- te))]
    batches = [(r = at_edge(te); zt = truth_at(z0, dd, te, events.t_add); sd = sqrt.(diag(r.z_cov));
                Dict("t_h" => te / 3600, "z" => r.z, "sd" => sd, "err_sd" => (r.z .- zt) ./ sd, "z_true" => zt,
                     "err_sd_graphfilter" => (r.z .- zt) ./ sd, "err_sd_system" => (r.z .- zt) ./ sd)) for te in edges]
    push!(cooks, Dict("seed" => seed, "z_by_hour" => [truth_at(z0, dd, h * 3600.0, events.t_add; from=true) for h in 0:29], "batches" => batches,
                      "temps" => temps, "wall_filter_s" => wall, "n_updates" => length(out), "n_uncommitted" => count(r -> !r.committed, out),
                      "als" => feed === nothing ? nothing : als_summary(feed.acc)))
    e = abs.(batches[end]["err_sd"])
    ep = [(r["pit"][1] - r["pit"][3]) / r["pit"][2] for r in temps]; ec = [(r["centre"][1] - r["centre"][3]) / r["centre"][2] for r in temps]; es = [(r["surface"][1] - r["surface"][3]) / r["surface"][2] for r in temps]
    nis_r = [r["nis"] / max(r["dof"], 1) for r in temps if r["dof"] > 0]
    walls = [r["wall_s"] for r in temps]
    @printf("cook %d (seed %d): at %.0f h |err|/sd max %.2f (%s) median %.2f; the temperatures' |err|/sd median pit %.2f centre %.2f surface %.2f; NIS/dof mean %.2f, NEES/16 mean %.2f, NEES_T/3 mean %.2f; %d arrivals (%d not committed), %.0f s, %.1f s per arrival (max %.0f)\n",
            c, seed, edges[end] / 3600, maximum(e), allnames[argmax(e)], sort(e)[div(end + 1, 2)],
            sort(abs.(ep))[div(end + 1, 2)], sort(abs.(ec))[div(end + 1, 2)], sort(abs.(es))[div(end + 1, 2)],
            sum(nis_r) / length(nis_r), sum(r["nees_z16"] for r in temps) / 16 / length(temps), sum(r["nees_T"] for r in temps) / 3 / length(temps),
            length(out), count(r -> !r.committed, out), wall, sum(walls) / length(walls), maximum(walls))
    for nm in ("u_leak", "k_e", "h_c", "r_max")
        i = idx[nm]
        @printf("    %-7s truth %s | smoother %s\n", nm, join([@sprintf("%+.2f", b["z_true"][i]) for b in batches], " "), join([@sprintf("%+.2f", b["err_sd"][i]) for b in batches], " "))
    end
    gm(g) = (v = [r["nees_g"][g] for r in temps if isfinite(r["nees_g"][g])]; isempty(v) ? NaN : sum(v) / length(v))
    println("    the state by group, NEES/dof mean: ", join([@sprintf("%s %.2f", g, gm(g)) for g in group_names], ", "))
    flush(stdout)
    save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
