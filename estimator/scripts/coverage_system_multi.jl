# The consistency test of the FACTOR-GRAPH backend on a TWO-PIECE cook (docs/estimator-consistency-roadmap.md,
# section 5): the plant a `CookSystem` of a flat and a bone-in rack on one pit solved by `system_solve` at truths
# drawn from the pilot's prior per seed (every piece's sites, the pit's, the areas, the flat's probe depth), each
# piece's events at its own time (the flat wrapped and pulled at the real-cook mock's times with its spritz train,
# the rack earlier, the add for both), read by the pit probe, a probe in each piece (the flat's at a latent depth,
# the rack's at its centre), the censored humidity sensor and one scale under both, each at the real-cook
# instruments' rate and AR(1) noise, every channel dark while the lid is open; the smoother over the same system
# (the pieces in the same order, or the other: `order 1`, the beliefs reported by name either way), scored as
# coverage_system.jl scores the one-piece bench: the latents against the truth at every batch edge, the pit and
# each piece's centre and surface against the plant's at every arrival, every state group of every piece, the
# normalized innovation squared, the standardized innovations per sensor. A sensor can be dropped (`drop core_rack`)
# for the missing-channel configuration. The record's schema is the one-piece worker's with per-piece groups and
# per-piece temperatures; bench_system.jl pools it.
#
# Run:  julia --project=estimator estimator/scripts/coverage_system_multi.jl [prior.json] [n_cooks] [seed0] [out.json] [edges_to_h]
#                 [nx 20] [calibration.json|default] [lag 0] [stale_ok 0/1] [als_lag_h 2] [order 0/1] [drop none|<sensor>]
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
n_cooks = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
seed0 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
out_name = length(ARGS) >= 4 ? ARGS[4] : "coverage_system_multi_$(seed0).json"
edges_to_h = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 14.0
NX = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 20
cal_file = length(ARGS) >= 7 ? ARGS[7] : "default"
LAG = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 0
stale_ok = length(ARGS) >= 9 ? ARGS[9] == "1" : false
als_lag_h = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 2.0
order_flip = length(ARGS) >= 11 ? ARGS[11] == "1" : false
drop = length(ARGS) >= 12 ? ARGS[12] : "none"

m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
prior_json = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
prior_phys = read_sites(prior_json["prior"]); phys_names = String.(prior_json["names"])
ev = m["events"]
T0 = Float64(meta["tcore0_K"])
flat = Slab1D(:flat; cfg=MeatConfig(; nx=NX, T0))
rack = Slab1D(:rack; cfg=MeatConfig(; nx=max(NX - 4, 6), geom_m=1.0, bone_frac=0.25, T0=T0 + 1.0, A_meat=0.05))
pieces_plant = [flat, rack]
pieces_model = order_flip ? [rack, flat] : [flat, rack]
sys_plant = CookSystem(pieces_plant; pit)
sys_model = CookSystem(pieces_model; pit)
blk_plant = system_latents(sys_plant, prior_phys, phys_names; probes=[:flat])
blk = system_latents(sys_model, prior_phys, phys_names; probes=[:flat])
names_plant = blk_plant.names; allnames = blk.names
idx = Dict(nm => i for (i, nm) in enumerate(allnames))
# the events: the flat at the real-cook mock's times with its spritz train, the rack (thinner) wrapped and pulled earlier, the add for both
t_wrap_flat, t_pull_flat, t_add = Float64(ev["t_wrap"]), Float64(ev["t_pull"]), Float64(ev["t_add"])
t_wrap_rack, t_pull_rack = 6.0 * 3600, 9.5 * 3600
cook_events = [CookEvent(:wrap, :flat, t_wrap_flat), CookEvent(:pull, :flat, t_pull_flat), CookEvent(:add, :pit, t_add),
               CookEvent(:spritz, :flat, Float64(ev["t_spritz"]); dt=Float64(ev["dt_spritz"]), m=Float64(ev["m_spritz"])),
               CookEvent(:wrap, :rack, t_wrap_rack), CookEvent(:pull, :rack, t_pull_rack)]
lid_events = (t_wrap=t_wrap_flat, t_pull=t_pull_flat, t_add=t_add, t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
lid_events_rack = (t_wrap=t_wrap_rack, t_pull=t_pull_rack)
edges = unique(vcat(3600.0 .* collect(2:2:floor(Int, edges_to_h)), [3600.0 * edges_to_h]))
t_end = edges[end]
nc = RealCookNoise()
# the instruments: the real-cook noise per sensor kind; the rack's probe like the flat's
SENSOR_IDS = (:pit, :core_flat, :core_rack, :humidity, :mass)
noise_of = Dict(:pit => (sd=nc.pit_sd, tau=nc.pit_tau, step=nc.pit_step, white=0.1), :core_flat => (sd=nc.core_sd, tau=nc.core_tau, step=nc.core_step, white=0.1),
                :core_rack => (sd=nc.core_sd, tau=nc.core_tau, step=nc.core_step, white=0.1), :humidity => (sd=nc.hum_sd, tau=nc.hum_tau, step=nc.hum_step, white=10.0),
                :mass => (sd=nc.mass_sd, tau=nc.mass_tau, step=nc.mass_step, white=0.005))
observable_of = Dict(:pit => PitTemperature(), :core_flat => TemperatureAt(:flat; depth_latent=Symbol("flat.core_depth_fraction")), :core_rack => TemperatureAt(:rack; s=0.0),
                     :humidity => ChamberVapor(), :mass => TotalMass())
sn, cal_sensors = cal_file == "default" ? (StateNoise(), nothing) :
                  load_system_calibration(isabspath(cal_file) ? cal_file : joinpath(REPO, "data", "calibration", cal_file); strict=!stale_ok)
sensor_noise = Dict(id => (white_sd=v.white, ar_sd=v.sd, tau=v.tau) for (id, v) in noise_of)
if cal_sensors !== nothing && !isempty(cal_sensors)
    for (id, v) in cal_sensors; haskey(sensor_noise, id) && (sensor_noise[id] = v); end
end
active_ids = [id for id in SENSOR_IDS if string(id) != drop]
drop == "none" || drop in string.(SENSOR_IDS) || error("no sensor $drop to drop (the sensors are $(join(string.(SENSOR_IDS), ", ")))")
sensors = [SensorSpec(id, observable_of[id], NoiseModel(; white_sd=sensor_noise[id].white_sd, ar_sd=sensor_noise[id].ar_sd, tau=sensor_noise[id].tau,
                                                      cap=(id === :humidity ? nc.hum_cap : Inf))) for id in active_ids]
fingerprint = backend_fingerprint()
bl = belief_layout(sys_model)
group_names = [E._als_group_name(bl, g) for (g, _) in bl.groups]
group_ranges = [rg for (_, rg) in bl.groups]
is_field(g) = occursin('.', g) && !endswith(g, ".bark")
field_basis = Dict(g => E.field_mode_basis(bl, gfull) for (g, (gfull, _)) in zip(group_names, bl.groups))      # the bulk and the smooth modes of a slab field
group_dof = Dict{String,Int}(g => (is_field(g) && length(rg) > 1 && field_basis[g] !== nothing ? size(field_basis[g], 2) : length(rg)) for (g, rg) in zip(group_names, group_ranges))
ip = first(E.belief_index(bl, :pit, :T_pit))
piece_ids = [pc.id for pc in pieces_model]
rT = Dict(id => E.belief_index(bl, id, :T) for id in piece_ids)
ia0 = idx["a0"]; keep16 = [i for i in 1:length(allnames) if i != ia0]
PIT_ICS_FIELDS = (:Tpit0, :mfuel0, :mpan0, :Tpan0, :Tfb0)

cooks = Any[]; skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    write(io, json_document(Dict("plant" => "a CookSystem of a flat (nx $NX) and a bone-in rack on one pit, system_solve at the truth; real-cook instruments per sensor",
                                 "backend" => "system", "estimator" => "the square-root fixed-lag smoother over the same CookSystem (online.jl), lag $LAG, the pieces " * (order_flip ? "in the other order" : "in the plant's order"),
                                 "names" => allnames, "prior_file" => prior_file, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped,
                                 "drift" => Dict("on" => false), "q" => zeros(length(allnames)), "latents_in_window" => "static",
                                 "state_noise" => state_noise_record(bl, sn), "state_noise_file" => cal_file,
                                 "sensors" => Dict(string(id) => Dict("white_sd" => v.white_sd, "ar_sd" => v.ar_sd, "tau" => v.tau) for (id, v) in sensor_noise if id in active_ids),
                                 "dropped" => drop, "noise_mode" => "matched", "pieces" => string.(piece_ids), "order_flipped" => order_flip,
                                 "physics" => fingerprint, "physics_only" => physics_fingerprint(), "adapt_level" => false, "groups" => group_names, "group_dof" => group_dof,
                                 "n_sigma" => 0, "lag" => LAG, "stale_ok" => stale_ok, "als_lag_h" => als_lag_h, "channels" => Dict(string(i) => string(id) for (i, id) in enumerate(SENSOR_IDS)),
                                 "events" => Dict("flat" => Dict("t_wrap_h" => t_wrap_flat / 3600, "t_pull_h" => t_pull_flat / 3600), "rack" => Dict("t_wrap_h" => t_wrap_rack / 3600, "t_pull_h" => t_pull_rack / 3600), "t_add_h" => t_add / 3600))))
end
println("the factor-graph backend on two pieces: prior ", prior_file, "; ", n_cooks, " cooks from seed ", seed0, "; edges to ", edges_to_h, " h; nx ", NX, "; lag ", LAG,
        "; the pieces ", order_flip ? "flipped" : "in order", "; dropped ", drop, "; the state noise from ", cal_file, "; backend ", fingerprint,
        als_lag_h > 0 ? "; the innovation autocovariances accumulated to $(als_lag_h) h" : "", "; record ", out_name)
flush(stdout)

"The plant's state at save index `k` as the model's belief state (the pieces by id) and its coordinates."
function truth_belief(o, k)
    h = system_handoff(o, k)
    by_id = Dict(pc.id => h.pieces[i] for (i, pc) in enumerate(pieces_plant))
    p0, play = pack_params(sys_model, [zeros(8) for _ in pieces_model], [pc.cfg.A_meat for pc in pieces_model], vcat(zeros(6), 0.5), u[1:2]; events=cook_events)
    pit_ics = merge(h.pit, (ignited=clamp(h.ignited, 1e-6, 1 - 1e-6),))
    x_sys = initial_state(sys_model, p0, play; ics=[by_id[pc.id] for pc in pieces_model], pit_ics)
    xb = belief_from_system(x_sys, bl)
    return xb, belief_encode(xb, bl)
end
function nees(e, C)
    F = cholesky(Matrix(Symmetric(C)); check=false)
    issuccess(F) ? dot(e, F \ e) : NaN
end
function nees_corr(e, C)
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
"Whether the lid is open at `τ` under either piece's events."
lid_open(τ) = lid_open_at(τ, lid_events, nc) || lid_open_at(τ, lid_events_rack, nc)
# the plant's own instruments: one unit AR(1) draw per sensor at its correlation time, drawn once per cook
function ar1_unit(rng, τ)
    ρ = exp(-dt / τ); a = zeros(length(t)); a[1] = randn(rng)
    for k in 2:length(t); a[k] = ρ * a[k - 1] + sqrt(1 - ρ^2) * randn(rng); end
    return a
end

for c in 1:n_cooks
    seed = seed0 + c - 1
    rng = MersenneTwister(3000 + seed)
    z_plant = draw_truth(rng, blk_plant.sites)                                           # in the plant's block order
    z_true = [z_plant[findfirst(==(nm), names_plant)] for nm in allnames]                 # the same truth in the model's order, by name
    t0 = time()
    p_true, play_true = pack_params(sys_plant, [zeros(8) for _ in pieces_plant], [pc.cfg.A_meat for pc in pieces_plant], vcat(zeros(6), 0.5), u; events=cook_events)
    p_true = apply_latents(blk_plant, z_plant, p_true, play_true)
    o = try
        system_solve(sys_plant, p_true, t, t)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the plant failed (%s); skipped\n", c, seed, sprint(showerror, err)[1:min(end, 100)]); push!(skipped, seed); save(); continue
    end
    all(isfinite, o.Tpit) || (@printf("cook %d (seed %d): the plant is not finite; skipped\n", c, seed); push!(skipped, seed); save(); continue)
    clog0 = CookLog(t, u; events=cook_events, sensors)
    bl_plant = belief_layout(sys_plant)
    hs = Dict(s.id => compile_observable(s.observable, sys_plant, bl_plant, blk_plant, clog0) for s in sensors)
    units = Dict(id => ar1_unit(rng, noise_of[id].tau) for id in SENSOR_IDS)              # every sensor's draw, dropped or not: the same plant reads whichever is present
    k_end = searchsortedlast(t, t_end + 1e-9)
    recs = MeasurementRecord[]
    for k in 1:k_end
        lid_open(t[k]) && continue
        for s in sensors
            id = s.id; nz = noise_of[id]
            (k - 1) % max(1, Int(round(nz.step / dt))) == 0 || continue
            clean = hs[id](o.X[:, k], z_plant, p_true, t, t[k])
            sd = id === :pit ? pit_sd_at(nc, t[k], t_add) : nz.sd
            v = clean + sd * units[id][k]
            id === :humidity && (v = min(v, nc.hum_cap))
            push!(recs, MeasurementRecord(id, t[k], v))
        end
    end
    arrivals = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
    clog = CookLog(t, u; events=cook_events, sensors, measurements=recs)
    model = SystemModel(sys_model, clog, blk; sn)
    feed = als_lag_h > 0 ? ALSFeed(model; lag_s=als_lag_h * 3600, bin_s=dt) : nothing
    sm = online_init(model; lag=LAG, als=feed)
    out = NamedTuple[]
    failed = nothing
    for grp in arrivals
        r = try
            assimilate!(sm, grp)
        catch err
            err isa InterruptException && rethrow()
            failed = err; break
        end
        push!(out, r)
    end
    if failed !== nothing
        @printf("cook %d (seed %d): the smoother failed at %.2f h (%s); skipped\n", c, seed, isempty(out) ? 0.0 : out[end].t / 3600, sprint(showerror, failed)[1:min(end, 120)])
        push!(skipped, seed); save(); continue
    end
    wall = time() - t0
    kk(te) = searchsortedlast(t, te + 1e-9)
    at_edge(te) = out[argmin(abs.([r.t for r in out] .- te))]
    batches = [(r = at_edge(te); sd = sqrt.(diag(r.z_cov));
                Dict("t_h" => te / 3600, "z" => r.z, "sd" => sd, "err_sd" => (r.z .- z_true) ./ sd, "z_true" => z_true,
                     "err_sd_graphfilter" => (r.z .- z_true) ./ sd, "err_sd_system" => (r.z .- z_true) ./ sd)) for te in edges]
    temps = Any[]
    for r in out
        k = kk(r.t)
        ez = r.z .- z_true
        xb_true, y_true = truth_belief(o, k)
        pieces_rec = Dict{String,Any}()
        iT = [ip]; eT = [r.T_pit[1] - o.Tpit[k]]
        for (i, pc) in enumerate(pieces_plant)
            id = pc.id; po = o.pieces[i]
            ic, is = first(rT[id]), last(rT[id])
            pieces_rec[string(id)] = Dict("centre" => [r.pieces[id].T_centre[1], r.pieces[id].T_centre[2], po.Tcore[k]], "surface" => [r.pieces[id].T_surface[1], r.pieces[id].T_surface[2], po.Tsurf[k]])
            append!(iT, [ic, is]); append!(eT, [r.pieces[id].T_centre[1] - po.Tcore[k], r.pieces[id].T_surface[1] - po.Tsurf[k]])
        end
        T_cov = r.x_cov[iT, iT]
        innov = Dict(string(findfirst(==(id), SENSOR_IDS)) => [v] for (id, v) in r.innov_std)
        flat_rec = pieces_rec["flat"]
        push!(temps, Dict("t_h" => r.t / 3600, "k" => k, "updated" => r.committed, "nees_g" => group_nees(r.y .- y_true, r.y_cov),
                          "pit" => [r.T_pit[1], r.T_pit[2], o.Tpit[k]], "centre" => flat_rec["centre"], "surface" => flat_rec["surface"], "pieces" => pieces_rec,
                          "nis" => r.nis, "dof" => r.dof, "innov" => innov,
                          "nees_z" => nees(ez, r.z_cov), "nees_z16" => nees(ez[keep16], r.z_cov[keep16, keep16]), "nees_T" => nees(eT, T_cov), "dof_T" => length(iT),
                          "level" => ones(length(SENSOR_IDS)), "faulty" => falses(length(SENSOR_IDS)),
                          "noise" => [[sensor_noise[id].ar_sd, exp(-dt / sensor_noise[id].tau)] for id in SENSOR_IDS], "wall_s" => r.wall,
                          "iters" => r.iters, "rounds" => r.attempts, "converged" => r.converged, "committed" => r.committed, "gradient" => r.gradient, "n_nodes" => r.n_nodes))
    end
    push!(cooks, Dict("seed" => seed, "z_by_hour" => [z_true for _ in 0:29], "batches" => batches, "temps" => temps, "wall_filter_s" => wall,
                      "n_updates" => length(out), "n_uncommitted" => count(r -> !r.committed, out), "als" => feed === nothing ? nothing : als_summary(feed.acc)))
    e = abs.(batches[end]["err_sd"])
    nis_r = [r["nis"] / max(r["dof"], 1) for r in temps if r["dof"] > 0]
    walls = [r["wall_s"] for r in temps]
    @printf("cook %d (seed %d): at %.0f h |err|/sd max %.2f (%s) median %.2f; NIS/dof mean %.2f, NEES/%d mean %.2f, NEES_T/%d mean %.2f; %d arrivals (%d not committed), %.0f s, %.1f s per arrival (max %.0f)\n",
            c, seed, edges[end] / 3600, maximum(e), allnames[argmax(e)], sort(e)[div(end + 1, 2)], sum(nis_r) / length(nis_r), length(keep16),
            sum(r["nees_z16"] for r in temps) / length(keep16) / length(temps), temps[1]["dof_T"], sum(r["nees_T"] for r in temps) / temps[1]["dof_T"] / length(temps),
            length(out), count(r -> !r.committed, out), wall, sum(walls) / length(walls), maximum(walls))
    gm(g) = (v = [r["nees_g"][g] for r in temps if isfinite(r["nees_g"][g])]; isempty(v) ? NaN : sum(v) / length(v))
    println("    the state by group, NEES/dof mean: ", join([@sprintf("%s %.2f", g, gm(g)) for g in group_names], ", "))
    flush(stdout)
    save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
