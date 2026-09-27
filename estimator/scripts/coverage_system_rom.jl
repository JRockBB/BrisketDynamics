# The consistency test of the FACTOR-GRAPH backend under a REDUCED-ORDER artifact (docs/estimator-consistency-roadmap.md,
# section 4): the plant the artifact's full-order piece (the half-thickness brick of train_rom_bench.jl) solved by
# `system_solve` at truths drawn from the artifact's own prior per seed, under the real-cook mock's damper and events;
# the estimator the artifact's slab with the artifact attached (every observation factor carrying the reduction
# bias and the widened noise; the artifact's process noise unless a calibration file gives one), read by the pit
# probe, a probe at the brick's centre, the censored humidity sensor and one scale at the real-cook instruments'
# rates and AR(1) noise, dark while the lid is open. Scored as the one-piece bench: the latents at the batch edges,
# the pit, the centre (the brick's core node) and the surface (the brick's exposed faces' mean) at every arrival, the
# state groups (the pit's scalars against the brick's; the slab's fields on their bulk against the brick's core line's),
# NIS and the standardized innovations. The record's schema is the one-piece worker's; bench_system.jl pools it.
#
# Run:  julia --project=estimator estimator/scripts/coverage_system_rom.jl [prior.json] [n_cooks] [seed0] [out.json] [edges_to_h]
#                 [rom.json] [calibration.json|default] [lag 0] [stale_ok 0/1] [als_lag_h 2]
#   rom.json: the artifact (data/calibration/rom_<name>.json, or an absolute path); its slab and brick are rebuilt from the
#             training script's geometry rule (the prior's central half-thickness, the flat's footprint, nx from the artifact)
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
out_name = length(ARGS) >= 4 ? ARGS[4] : "coverage_system_rom_$(seed0).json"
edges_to_h = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 14.0
rom_file = length(ARGS) >= 6 ? ARGS[6] : "rom_halfbrick_flat.json"
cal_file = length(ARGS) >= 7 ? ARGS[7] : "default"
LAG = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : 0
stale_ok = length(ARGS) >= 9 ? ARGS[9] == "1" : false
als_lag_h = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : 2.0

m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
prior_json = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
prior_phys = read_sites(prior_json["prior"]); phys_names = String.(prior_json["names"])
ev = m["events"]
rom = load_rom(isabspath(rom_file) ? rom_file : joinpath(REPO, "data", "calibration", rom_file); strict=!stale_ok)
# the geometry as the training script built it, the slab's mesh from the artifact
L_fom = from_z(z_moments(prior_phys[4])[1], prior_phys[4])
Ly, Lz = 0.30, 0.40
NX = Int(rom.rom["nx"])
cfg = MeatConfig(; nx=NX, n_shrink=0.0, T0=Float64(meta["tcore0_K"]), A_meat=Ly * Lz)
dims = Int.(rom.fom["dims"])
fom = Brick3D(:b; Lx=L_fom, Ly, Lz, nx=dims[1], ny=dims[2], nz=dims[3], exposed=(false, true, true, true, true, true), cfg)
rom_piece = Slab1D(:b; cfg)
sys_fom = CookSystem(fom; pit); sys_rom = CookSystem(rom_piece; pit)
prior_rom = copy(prior_phys)
prior_rom[4] = PriorSite("lognormal", log(L_fom), prior_phys[4].p2)
blk = system_latents(sys_rom, prior_rom, phys_names; area=false)
allnames = blk.names
idx = Dict(nm => i for (i, nm) in enumerate(allnames))
t_wrap, t_add, t_spritz, dt_spritz, m_spritz = Float64(ev["t_wrap"]), Float64(ev["t_add"]), Float64(ev["t_spritz"]), Float64(ev["dt_spritz"]), Float64(ev["m_spritz"])
t_pull = Float64(ev["t_pull"])
cook_events = [CookEvent(:wrap, :b, t_wrap), CookEvent(:pull, :b, t_pull), CookEvent(:add, :pit, t_add), CookEvent(:spritz, :b, t_spritz; dt=dt_spritz, m=m_spritz)]
lid_events = (t_wrap=t_wrap, t_pull=t_pull, t_add=t_add, t_spritz=t_spritz, dt_spritz=dt_spritz, m_spritz=m_spritz)
edges = unique(vcat(3600.0 .* collect(2:2:floor(Int, edges_to_h)), [3600.0 * edges_to_h]))
t_end = edges[end]
nc = RealCookNoise()
SENSOR_IDS = (:pit, :core, :humidity, :mass)
noise_of = Dict(:pit => (sd=nc.pit_sd, tau=nc.pit_tau, step=nc.pit_step, white=0.1), :core => (sd=nc.core_sd, tau=nc.core_tau, step=nc.core_step, white=0.1),
                :humidity => (sd=nc.hum_sd, tau=nc.hum_tau, step=nc.hum_step, white=10.0), :mass => (sd=nc.mass_sd, tau=nc.mass_tau, step=nc.mass_step, white=0.005))
observable_of = Dict(:pit => PitTemperature(), :core => TemperatureAt(:b; s=0.0), :humidity => ChamberVapor(), :mass => TotalMass())
sn_cal, cal_sensors = cal_file == "default" ? (nothing, nothing) :
                      load_system_calibration(isabspath(cal_file) ? cal_file : joinpath(REPO, "data", "calibration", cal_file); strict=!stale_ok)
sensor_noise = Dict(id => (white_sd=v.white, ar_sd=v.sd, tau=v.tau) for (id, v) in noise_of)
if cal_sensors !== nothing && !isempty(cal_sensors)
    for (id, v) in cal_sensors; haskey(sensor_noise, id) && (sensor_noise[id] = v); end
end
sensors = [SensorSpec(id, observable_of[id], NoiseModel(; white_sd=sensor_noise[id].white_sd, ar_sd=sensor_noise[id].ar_sd, tau=sensor_noise[id].tau,
                                                      cap=(id === :humidity ? nc.hum_cap : Inf))) for id in SENSOR_IDS]
model0 = SystemModel(sys_rom, CookLog(t, u; events=cook_events, sensors), blk; sn=sn_cal, rom)
sn = model0.sn                                                                          # the calibration's, else the artifact's own
fingerprint = backend_fingerprint()
bl = model0.bl
group_names = [E._als_group_name(bl, g) for (g, _) in bl.groups]
group_ranges = [rg for (_, rg) in bl.groups]
is_field(g) = occursin('.', g) && !endswith(g, ".bark")
group_dof = Dict{String,Int}(g => (is_field(g) && length(rg) > 1 ? 1 : length(rg)) for (g, rg) in zip(group_names, group_ranges))
ip = first(E.belief_index(bl, :pit, :T_pit)); rT = E.belief_index(bl, :b, :T); ic, is = first(rT), last(rT)
ia0 = idx["a0"]; keep16 = [i for i in 1:length(allnames) if i != ia0]
field_of = Dict("T" => :T, "W" => :W, "C" => :C, "F" => :F, "S" => :S)

cooks = Any[]; skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    write(io, json_document(Dict("plant" => "the artifact's full-order brick ($(fom.mesh.N) nodes, top and sides exposed), system_solve at the truth; real-cook instruments",
                                 "backend" => "system", "estimator" => "the square-root fixed-lag smoother over the artifact's slab (nx $NX) with the artifact $(rom.archetype) attached (online.jl), lag $LAG",
                                 "rom" => Dict("file" => rom_file, "archetype" => rom.archetype, "fingerprint" => rom.fingerprint),
                                 "names" => allnames, "prior_file" => prior_file, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped,
                                 "drift" => Dict("on" => false), "q" => zeros(length(allnames)), "latents_in_window" => "static",
                                 "state_noise" => state_noise_record(bl, sn), "state_noise_file" => cal_file,
                                 "sensors" => Dict(string(id) => Dict("white_sd" => v.white_sd, "ar_sd" => v.ar_sd, "tau" => v.tau) for (id, v) in sensor_noise), "noise_mode" => "matched",
                                 "physics" => fingerprint, "physics_only" => physics_fingerprint(), "adapt_level" => false, "groups" => group_names, "group_dof" => group_dof,
                                 "n_sigma" => 0, "lag" => LAG, "stale_ok" => stale_ok, "als_lag_h" => als_lag_h,
                                 "channels" => Dict("1" => "pit probe", "2" => "core probe", "3" => "humidity", "4" => "scale"))))
end
println("the factor-graph backend under the artifact ", rom.archetype, " (", rom.fingerprint, "): prior ", prior_file, "; ", n_cooks, " cooks from seed ", seed0, "; edges to ", edges_to_h,
        " h; the brick ", fom.mesh.N, " nodes, the slab nx ", NX, "; lag ", LAG, "; the state noise ", cal_file == "default" ? "the artifact's" : "from $cal_file", "; backend ", fingerprint,
        als_lag_h > 0 ? "; the innovation autocovariances accumulated to $(als_lag_h) h" : "", "; record ", out_name)
flush(stdout)

function nees(e, C)
    F = cholesky(Matrix(Symmetric(C)); check=false)
    issuccess(F) ? dot(e, F \ e) : NaN
end
function nees_corr(e, C)
    s = sqrt.(max.(diag(C), 1e-300))
    F = cholesky(Matrix(Symmetric(C ./ (s * s'))); check=false)
    issuccess(F) ? dot(e ./ s, F \ (e ./ s)) : NaN
end
"The slab belief's state groups against the brick: the pit's scalars node for node; a field on its bulk (the belief's node mean in the coordinates against the brick's core line's mean, encoded); the bark's scalars against the brick's exposed faces' means."
function group_nees(r, o, k)
    x_fom = o.X[:, k]
    out = Dict{String,Float64}()
    for (g, rg) in zip(group_names, group_ranges)
        if occursin('.', g)
            f = String(last(rsplit(g, '.'; limit=2)))
            if haskey(field_of, f)
                line = project_to_line(rom, fom, x_fom[state_index(sys_fom.layout, :b, field_of[f])])
                kind = bl.kinds[first(rg)]
                y_true = E._belief_enc(sum(line) / length(line), kind)
                v = sum(r.y_cov[rg, rg]) / length(rg)^2
                out[g] = (sum(r.y[rg]) / length(rg) - y_true)^2 / v
            else                                                                       # the bark, smoke and ring: the brick's exposed faces' means against the slab's scalars
                w = last.(fom.mesh.boundary) ./ fom.mesh.area; nodes = first.(fom.mesh.boundary)
                truth = [dot(w, x_fom[state_index(sys_fom.layout, :b, fld)][nodes]) for fld in (:bark_b, :bark_s)]
                rb = rg[1:2]                                                           # the ring has no counterpart on the brick
                e = [E._belief_enc(truth[j], bl.kinds[rb[j]]) for j in 1:2] .- r.y[rb]
                out[g] = nees_corr(e, r.y_cov[rb, rb]) / 2
            end
        else
            e = [E._belief_enc(x_fom[state_index(sys_fom.layout, :pit, Symbol(g))][1], bl.kinds[first(rg)])] .- r.y[rg]
            out[g] = nees_corr(e, r.y_cov[rg, rg])
        end
    end
    return out
end
function ar1_unit(rng, τ)
    ρ = exp(-dt / τ); a = zeros(length(t)); a[1] = randn(rng)
    for k in 2:length(t); a[k] = ρ * a[k - 1] + sqrt(1 - ρ^2) * randn(rng); end
    return a
end
p_pack, play_pack = pack_params(sys_fom, [zeros(8)], [1.0], vcat(zeros(6), 0.5), u; events=cook_events)

for c in 1:n_cooks
    seed = seed0 + c - 1
    rng = MersenneTwister(4000 + seed)
    z_true = draw_truth(rng, blk.sites)
    t0 = time()
    p_true = apply_latents(blk, z_true, p_pack, play_pack)
    o = try
        system_solve(sys_fom, p_true, t, t)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the plant failed (%s); skipped\n", c, seed, sprint(showerror, err)[1:min(end, 100)]); push!(skipped, seed); save(); continue
    end
    all(isfinite, o.Tpit) || (@printf("cook %d (seed %d): the plant is not finite; skipped\n", c, seed); push!(skipped, seed); save(); continue)
    bl_fom = belief_layout(sys_fom)
    clog0 = CookLog(t, u; events=cook_events, sensors)
    hs = Dict(s.id => compile_observable(s.observable, sys_fom, bl_fom, blk, clog0) for s in sensors)
    units = Dict(id => ar1_unit(rng, noise_of[id].tau) for id in SENSOR_IDS)
    k_end = searchsortedlast(t, t_end + 1e-9)
    recs = MeasurementRecord[]
    for k in 1:k_end
        lid_open_at(t[k], lid_events, nc) && continue
        for s in sensors
            id = s.id; nz = noise_of[id]
            (k - 1) % max(1, Int(round(nz.step / dt))) == 0 || continue
            clean = hs[id](o.X[:, k], z_true, p_true, t, t[k])
            sd = id === :pit ? pit_sd_at(nc, t[k], t_add) : nz.sd
            v = clean + sd * units[id][k]
            id === :humidity && (v = min(v, nc.hum_cap))
            push!(recs, MeasurementRecord(id, t[k], v))
        end
    end
    arrivals = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
    clog = CookLog(t, u; events=cook_events, sensors, measurements=recs)
    model = SystemModel(sys_rom, clog, blk; sn, rom)
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
    po = o.pieces[1]
    temps = Any[]
    for r in out
        k = kk(r.t)
        ez = r.z .- z_true
        eT = [r.T_pit[1] - o.Tpit[k], r.pieces[:b].T_centre[1] - po.Tcore[k], r.pieces[:b].T_surface[1] - po.Tsurf[k]]
        T_cov = r.x_cov[[ip, ic, is], [ip, ic, is]]
        innov = Dict(string(findfirst(==(id), SENSOR_IDS)) => [v] for (id, v) in r.innov_std)
        push!(temps, Dict("t_h" => r.t / 3600, "k" => k, "updated" => r.committed, "nees_g" => group_nees(r, o, k),
                          "pit" => [r.T_pit[1], r.T_pit[2], o.Tpit[k]], "centre" => [r.pieces[:b].T_centre[1], r.pieces[:b].T_centre[2], po.Tcore[k]],
                          "surface" => [r.pieces[:b].T_surface[1], r.pieces[:b].T_surface[2], po.Tsurf[k]],
                          "nis" => r.nis, "dof" => r.dof, "innov" => innov,
                          "nees_z" => nees(ez, r.z_cov), "nees_z16" => nees(ez[keep16], r.z_cov[keep16, keep16]), "nees_T" => nees(eT, T_cov), "dof_T" => 3,
                          "level" => ones(length(SENSOR_IDS)), "faulty" => falses(length(SENSOR_IDS)),
                          "noise" => [[sensor_noise[id].ar_sd, exp(-dt / sensor_noise[id].tau)] for id in SENSOR_IDS], "wall_s" => r.wall,
                          "iters" => r.iters, "rounds" => r.attempts, "converged" => r.converged, "committed" => r.committed, "gradient" => r.gradient, "n_nodes" => r.n_nodes))
    end
    push!(cooks, Dict("seed" => seed, "z_by_hour" => [z_true for _ in 0:29], "batches" => batches, "temps" => temps, "wall_filter_s" => wall,
                      "n_updates" => length(out), "n_uncommitted" => count(r -> !r.committed, out), "als" => feed === nothing ? nothing : als_summary(feed.acc)))
    e = abs.(batches[end]["err_sd"])
    ep = [(r["pit"][1] - r["pit"][3]) / r["pit"][2] for r in temps]; ec = [(r["centre"][1] - r["centre"][3]) / r["centre"][2] for r in temps]; es = [(r["surface"][1] - r["surface"][3]) / r["surface"][2] for r in temps]
    nis_r = [r["nis"] / max(r["dof"], 1) for r in temps if r["dof"] > 0]
    walls = [r["wall_s"] for r in temps]
    @printf("cook %d (seed %d): at %.0f h |err|/sd max %.2f (%s) median %.2f; the temperatures' |err|/sd median pit %.2f centre %.2f surface %.2f; NIS/dof mean %.2f, NEES/%d mean %.2f, NEES_T/3 mean %.2f; %d arrivals (%d not committed), %.0f s, %.1f s per arrival (max %.0f)\n",
            c, seed, edges[end] / 3600, maximum(e), allnames[argmax(e)], sort(e)[div(end + 1, 2)],
            sort(abs.(ep))[div(end + 1, 2)], sort(abs.(ec))[div(end + 1, 2)], sort(abs.(es))[div(end + 1, 2)],
            sum(nis_r) / length(nis_r), length(keep16), sum(r["nees_z16"] for r in temps) / length(keep16) / length(temps), sum(r["nees_T"] for r in temps) / 3 / length(temps),
            length(out), count(r -> !r.committed, out), wall, sum(walls) / length(walls), maximum(walls))
    gm(g) = (v = [r["nees_g"][g] for r in temps if isfinite(r["nees_g"][g])]; isempty(v) ? NaN : sum(v) / length(v))
    println("    the state by group, NEES/dof mean: ", join([@sprintf("%s %.2f", g, gm(g)) for g in group_names], ", "))
    flush(stdout)
    save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
