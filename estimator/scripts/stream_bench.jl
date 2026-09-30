# The factor-graph backend on the real configuration with STREAMING sensors: the real-cook mock's plants (truths drawn from
# the pilot's prior per seed, the monolith solved in segments, the 40-node slab, the spritz/wrap/add/pull events), each
# sensor reading on its own clock (its own interval with jitter, timestamps to the second, none while the lid is open,
# its OU discrepancy advanced exactly between its own irregular times, its white floor), the estimator updated as readings
# arrive at the live lag. Scored at every update against the plant at that time: the innovations' NIS against their
# predicted covariance, the latents' and the three temperatures' NEES, every state group's NEES, the temperatures'
# standardized errors; the latents at the two-hour edges. The record's schema is coverage_system.jl's.
#
# Run: julia --threads=4 --project=estimator estimator/scripts/stream_bench.jl [n_cooks] [seed0] [out.json] [hours] [lag]
#        [attach slice|node] [pace] [pit_s] [core_s] [hum_s] [mass_s] [node_ratio]
#   pace: 0 takes every arrival alone; k > 0 is compute-paced: an update takes every reading that arrived during the
#         previous update's measured compute time times k (k = 1 the live machine; k > 1 a slower one, fewer updates,
#         the same posterior at each: the window solve is exact however readings are grouped)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

A_ = ARGS
n_cooks = length(A_) >= 1 ? parse(Int, A_[1]) : 1
seed0 = length(A_) >= 2 ? parse(Int, A_[2]) : 1
out_name = length(A_) >= 3 ? A_[3] : "stream_bench_$(seed0).json"
hours = length(A_) >= 4 ? parse(Float64, A_[4]) : 14.0
LAG = length(A_) >= 5 ? parse(Int, A_[5]) : 3
ATTACH = length(A_) >= 6 ? Symbol(A_[6]) : :slice
PACE = length(A_) >= 7 ? parse(Float64, A_[7]) : 0.0
STEP = Dict(:pit => length(A_) >= 8 ? parse(Float64, A_[8]) : 30.0, :core => length(A_) >= 9 ? parse(Float64, A_[9]) : 30.0,
            :humidity => length(A_) >= 10 ? parse(Float64, A_[10]) : 120.0, :mass => length(A_) >= 11 ? parse(Float64, A_[11]) : 120.0)
NODE_RATIO = length(A_) >= 12 ? parse(Float64, A_[12]) : 1.0
JITTER = 0.25                                                    # each interval drawn in step × (1 ± 0.25)

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
PIECE = :brisket
sys = CookSystem(Slab1D(PIECE; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[PIECE])
sites, allnames = blk.sites, blk.names
idx = Dict(nm => i for (i, nm) in enumerate(allnames))
AREA, DEPTH = latent_index(blk, "A_meat"), latent_index(blk, "core_depth_fraction")
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
cook_events = [CookEvent(:wrap, PIECE, events.t_wrap), CookEvent(:pull, PIECE, events.t_pull), CookEvent(:add, :pit, events.t_add),
               CookEvent(:spritz, PIECE, events.t_spritz; dt=events.dt_spritz, m=events.m_spritz)]
t_end = 3600.0 * hours
edges = unique(vcat(3600.0 .* collect(2:2:floor(Int, hours)), [t_end]))
nc = RealCookNoise()
WHITE = Dict(:pit => 0.1, :core => 0.1, :humidity => 10.0, :mass => 0.005)
OU = Dict(:pit => (nc.pit_sd, nc.pit_tau), :core => (nc.core_sd, nc.core_tau), :humidity => (nc.hum_sd, nc.hum_tau), :mass => (nc.mass_sd, nc.mass_tau))
SENSOR_IDS = (:pit, :core, :humidity, :mass)
observable_of = Dict(:pit => PitTemperature(), :core => TemperatureAt(PIECE; depth_latent=:core_depth_fraction), :humidity => ChamberVapor(), :mass => TotalMass())
sensors = [SensorSpec(id, observable_of[id], NoiseModel(; white_sd=WHITE[id], ar_sd=OU[id][1], tau=OU[id][2], cap=(id === :humidity ? nc.hum_cap : Inf))) for id in SENSOR_IDS]
sn = StateNoise()
bl = belief_layout(sys)
group_names = [E._als_group_name(bl, g) for (g, _) in bl.groups]
group_ranges = [rg for (_, rg) in bl.groups]
ip = first(E.belief_index(bl, :pit, :T_pit)); rT = E.belief_index(bl, PIECE, :T); ic, is = first(rT), last(rT)
ia0 = idx["a0"]; keep16 = [i for i in 1:length(sites) if i != ia0]
p_pack, play_pack = pack_params(sys, [zeros(8)], [cfg.A_meat], vcat(zeros(6), 0.5), u[1:2]; events=cook_events)
PIT_ICS_FIELDS = (:Tpit0, :mfuel0, :mpan0, :Tpan0, :Tfb0)

"Each sensor's reading times: its own clock with jitter and a random phase, whole seconds, none while the lid is open."
function schedules(rng)
    Dict(id => begin
        ts = Float64[]; τ = round(rand(rng) * STEP[id])
        while τ <= t_end
            τ > 0 && !lid_open_at(τ, events, nc) && push!(ts, τ)
            τ += round(STEP[id] * (1 + JITTER * (2rand(rng) - 1)))
        end
        ts
    end for id in SENSOR_IDS)
end
"The plant in segments saving at every time in `tsave` (the damper on its own grid): the monolith's outputs at those times."
function plant(xs, seg_edges, tsave, A)
    bounds = vcat(1, collect(Int, seg_edges), length(t))
    outs = NamedTuple[]; ics = (;)
    for i in 1:(length(bounds) - 1)
        ia, ib = bounds[i], bounds[i + 1]
        tt = t[ia:ib]; uu = u[ia:ib]
        ts = sort(unique(vcat(tt[1], [τ for τ in tsave if tt[1] < τ < tt[end]], tt[end])))
        φ = xs[i][9:15]
        if i > 1
            ics, a0 = handoff(outs[end], size(outs[end].T_field, 2)); φ = vcat(φ[1:6], a0)
        end
        push!(outs, monolith_solve(xs[i][1:8], φ, uu, tt, ts; pit, cfg, A_meat=A, events..., ics...))
    end
    names = keys(outs[1])
    vals = map(names) do nm
        v1 = getfield(outs[1], nm)
        parts = [i == 1 ? getfield(o, nm) : E._drop_first(getfield(o, nm)) for (i, o) in enumerate(outs)]
        v1 isa AbstractMatrix && size(v1, 2) == length(outs[1].t) ? reduce(hcat, parts) :
        v1 isa AbstractVector && length(v1) == length(outs[1].t) ? reduce(vcat, parts) : getfield(outs[end], nm)
    end
    return NamedTuple{names}(vals)
end
function truth_belief(o, k)
    ics, a = handoff(o, k)
    pf = Tuple(f for f in keys(ics) if !(f in PIT_ICS_FIELDS))
    piece_ics = NamedTuple{pf}(Tuple(getfield(ics, f) for f in pf))
    pit_ics = (Tpit0=ics.Tpit0, mfuel0=ics.mfuel0, mpan0=ics.mpan0, Tpan0=ics.Tpan0, Tfb0=ics.Tfb0, ignited=clamp(a, 1e-6, 1 - 1e-6))
    x_sys = initial_state(sys, p_pack, play_pack; ics=[piece_ics], pit_ics)
    xb = belief_from_system(x_sys, bl)
    return xb, belief_encode(xb, bl)
end
nees(e, C) = (F = cholesky(Matrix(Symmetric(C)); check=false); issuccess(F) ? dot(e, F \ e) : NaN)
function nees_corr(e, C)
    s = sqrt.(max.(diag(C), 1e-300)); F = cholesky(Matrix(Symmetric(C ./ (s * s'))); check=false)
    issuccess(F) ? dot(e ./ s, F \ (e ./ s)) : NaN
end
# a field under :offset is scored on its bulk (the node mean, one dof); a scalar or bark group node by node
function group_nees(ey, Cy)
    Dict(g => begin
        if length(rg) > 1 && sn.fields === :offset && !endswith(g, "bark")
            a = fill(1 / length(rg), 1, length(rg)); nees(a * ey[rg], a * Cy[rg, rg] * a')
        else
            nees_corr(ey[rg], Cy[rg, rg]) / length(rg)
        end
    end for (g, rg) in zip(group_names, group_ranges))
end

cooks = Any[]; skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    write(io, json_document(Dict("plant" => "monolith (1-D, 40 nodes) in segments, constant parameters; streaming real-cook instruments",
        "backend" => "system", "estimator" => "the fixed-lag smoother, lag $LAG, attach $ATTACH, node_ratio $NODE_RATIO",
        "names" => allnames, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped, "steps_s" => Dict(string(k) => v for (k, v) in STEP),
        "jitter" => JITTER, "pace" => PACE, "groups" => group_names, "lag" => LAG, "backend_fingerprint" => backend_fingerprint())))
end
println("stream bench: ", n_cooks, " cooks from seed ", seed0, ", ", hours, " h, lag ", LAG, ", attach ", ATTACH, ", pace ", PACE,
        ", steps ", join(["$k $(v) s" for (k, v) in STEP], ", "), ", node_ratio ", NODE_RATIO); flush(stdout)

for c in 1:n_cooks
    seed = seed0 + c - 1
    rng = MersenneTwister(2000 + seed)
    z0 = draw_truth(rng, sites)
    dd0 = DriftDraw(rng, idx; sites)
    dd = DriftDraw(Dict(k => zero(v) for (k, v) in dd0.inc), Dict(k => 0.0 for (k, v) in dd0.jump))
    seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
    xs = [from_z(z, sites) for z in blocks]
    Am, depth = xs[1][AREA], probe_depth(xs[1][4], xs[1][DEPTH])
    rs = MersenneTwister(9000 + seed)
    sched = schedules(rs)
    alltimes = sort(unique(vcat(values(sched)...)))
    t0w = time()
    o = try
        plant(xs, seg_edges, alltimes, Am)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the plant failed; skipped\n", c, seed); push!(skipped, seed); save(); continue
    end
    kof = Dict(τ => k for (k, τ) in enumerate(o.t))
    # the readings: the clean value at the reading's time, the channel's OU advanced exactly between its own times, the white floor
    recs = MeasurementRecord[]
    for id in SENSOR_IDS
        d = 0.0; tp = -Inf
        sd0, τc = OU[id]
        for τ in sched[id]
            ρ = isfinite(tp) ? exp(-(τ - tp) / τc) : 0.0
            d = ρ * d + sqrt(1 - ρ^2) * randn(rs); tp = τ
            cl = clean_at(o, kof[τ]; A=Am, L=xs[1][4], depth, cfg)
            sd = id === :pit ? pit_sd_at(nc, τ, events.t_add) : sd0
            v = getfield(cl, id) + sd * d + WHITE[id] * randn(rs)
            id === :humidity && (v = min(v, nc.hum_cap))
            push!(recs, MeasurementRecord(id, τ, v))
        end
    end
    sort!(recs; by=r -> r.t)
    clog = CookLog(t, u; events=cook_events, sensors, measurements=recs)
    model = SystemModel(sys, clog, blk; sn)
    sm = online_init(model; lag=LAG, attach=ATTACH, node_ratio=NODE_RATIO)
    temps = Any[]; out = NamedTuple[]; failed = nothing
    i = 1; clock = 0.0
    while i <= length(recs)
        # the batch: this reading, and under compute pacing every reading that arrived while the last update ran
        j = i
        horizon = PACE > 0 ? max(recs[i].t, clock) : recs[i].t
        while j < length(recs) && recs[j + 1].t <= horizon + 1e-9; j += 1; end
        batch = recs[i:j]
        r = try
            assimilate!(sm, batch)
        catch err
            err isa InterruptException && rethrow(); failed = err; break
        end
        PACE > 0 && (clock = recs[j].t + PACE * r.wall)
        k = kof[r.t]
        zt = truth_at(z0, dd, r.t, events.t_add); ez = r.z .- zt
        _, y_true = truth_belief(o, k)
        eT = [r.T_pit[1] - o.Tpit[k], r.pieces[PIECE].T_centre[1] - o.Tcore[k], r.pieces[PIECE].T_surface[1] - o.Tsurf[k]]
        push!(temps, Dict("t_h" => r.t / 3600, "k" => k, "n_readings" => length(batch), "committed" => r.committed, "updated" => r.committed,
            "nees_g" => group_nees(r.y .- y_true, r.y_cov),
            "pit" => [r.T_pit[1], r.T_pit[2], o.Tpit[k]], "centre" => [r.pieces[PIECE].T_centre[1], r.pieces[PIECE].T_centre[2], o.Tcore[k]],
            "surface" => [r.pieces[PIECE].T_surface[1], r.pieces[PIECE].T_surface[2], o.Tsurf[k]],
            "nis" => r.nis, "dof" => r.dof, "innov" => Dict(string(findfirst(==(id), SENSOR_IDS)) => [v] for (id, v) in r.innov_std),
            "nees_z" => nees(ez, r.z_cov), "nees_z16" => nees(ez[keep16], r.z_cov[keep16, keep16]), "nees_T" => nees(eT, r.x_cov[[ip, ic, is], [ip, ic, is]]),
            "wall_s" => r.wall, "iters" => r.iters, "converged" => r.converged, "n_nodes" => r.n_nodes))
        push!(out, (t=r.t, z=copy(r.z), z_cov=copy(r.z_cov), committed=r.committed))
        if true
            @printf("  seed %d t %.2f h: %d updates, %d nodes in window, last update %.1f s (%d readings), NIS/dof %.2f\n",
                    seed, r.t / 3600, length(out), r.n_nodes, r.wall, length(batch), r.nis / max(r.dof, 1)); flush(stdout)
        end
        i = j + 1
    end
    if failed !== nothing
        @printf("cook %d (seed %d): the estimator failed at %.2f h (%s); skipped\n", c, seed, isempty(out) ? 0.0 : out[end].t / 3600, sprint(showerror, failed)[1:min(end, 160)])
        push!(skipped, seed); save(); continue
    end
    at_edge(te) = out[argmin(abs.([x.t for x in out] .- te))]
    batches = [(x = at_edge(te); zt = truth_at(z0, dd, te, events.t_add); sd = sqrt.(diag(x.z_cov));
                Dict("t_h" => te / 3600, "z" => x.z, "sd" => sd, "err_sd" => (x.z .- zt) ./ sd, "z_true" => zt,
                     "err_sd_graphfilter" => (x.z .- zt) ./ sd, "err_sd_system" => (x.z .- zt) ./ sd)) for te in edges]
    push!(cooks, Dict("seed" => seed, "batches" => batches, "temps" => temps, "wall_filter_s" => time() - t0w, "n_updates" => length(out),
                      "n_readings" => length(recs), "n_uncommitted" => count(x -> !x.committed, out)))
    zs(key) = [(x[key][1] - x[key][3]) / x[key][2] for x in temps]
    med(v) = sort(abs.(v))[div(end + 1, 2)]
    nis_r = [x["nis"] / x["dof"] for x in temps if x["dof"] > 0 && isfinite(x["nis"])]
    @printf("cook %d (seed %d): %d readings in %d updates (%d not committed), %.0f s; |err|/sd median pit %.2f centre %.2f surface %.2f; NIS/dof mean %.2f; NEES/16 mean %.2f; NEES_T/3 mean %.2f\n",
            c, seed, length(recs), length(out), count(x -> !x.committed, out), time() - t0w, med(zs("pit")), med(zs("centre")), med(zs("surface")),
            sum(nis_r) / max(length(nis_r), 1), sum(x["nees_z16"] for x in temps) / 16 / length(temps), sum(x["nees_T"] for x in temps) / 3 / length(temps))
    gm(g) = (v = [x["nees_g"][g] for x in temps if isfinite(x["nees_g"][g])]; isempty(v) ? NaN : sum(v) / length(v))
    println("    the state by group, NEES/dof mean: ", join([@sprintf("%s %.2f", g, gm(g)) for g in group_names], ", "))
    println("    the latents at the edges, err/sd: ", join([@sprintf("%.0fh max %.2f (%s)", b["t_h"], maximum(abs.(b["err_sd"])), allnames[argmax(abs.(b["err_sd"]))]) for b in batches], "; "))
    flush(stdout); save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
