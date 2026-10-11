# The factor-graph backend on the real configuration with STREAMING sensors: the real-cook mock's plants (truths drawn from
# the pilot's prior per seed, the monolith solved in segments, the 40-node slab, the spritz/wrap/add/pull events), each
# sensor reading on its own clock (its own interval with jitter, timestamps to the second, none while the lid is open,
# its OU discrepancy advanced exactly between its own irregular times, its white floor), the estimator updated as readings
# arrive at the live lag. Scored at every update against the plant at that time: the innovations' NIS against their
# predicted covariance, the latents' and the three temperatures' NEES, every state group's NEES, the temperatures'
# standardized errors; the latents at the two-hour edges. The record's schema is coverage_system.jl's.
#
# Run: julia --threads=4 --project=estimator estimator/scripts/stream_bench.jl [n_cooks] [seed0] [out.json] [hours] [lag]
#        [attach slice|node] [pace] [pit_s] [core_s] [hum_s] [mass_s] [node_ratio] [lag_max] [prior_relin] [slice_g_refresh]
#   pace: -1 is an offline node-batched diagnostic; 0 takes each arrival time; k > 0 is compute-paced: an update takes every reading that arrived during the
#         previous update's measured compute time times k (k = 1 the live machine; k > 1 a slower one, fewer updates,
#         readings retain their original timestamps). Node placement is independent of update scheduling.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf, Serialization
const E = BrisketEstimator

A_ = ARGS
# the defaults are the SHIPPED configuration (BrisketEstimator.SHIPPED_STREAM_CONFIG); arguments and env
# deviate from it only for explicit experiments, and the manifest records whatever actually ran
const SHIPPED = E.SHIPPED_STREAM_CONFIG
n_cooks = length(A_) >= 1 ? parse(Int, A_[1]) : 1
seed0 = length(A_) >= 2 ? parse(Int, A_[2]) : 1
out_name = length(A_) >= 3 ? A_[3] : "stream_bench_$(seed0).json"
hours = length(A_) >= 4 ? parse(Float64, A_[4]) : 14.0
LAG = length(A_) >= 5 ? parse(Int, A_[5]) : SHIPPED.lag
ATTACH = length(A_) >= 6 ? Symbol(A_[6]) : SHIPPED.attach
PACE = length(A_) >= 7 ? parse(Float64, A_[7]) : 0.0
UPDATE_POLICY = PACE < 0 ? "node_batches" : PACE == 0 ? "every_arrival" : "processing_backlog"
STEP = Dict(:pit => length(A_) >= 8 ? parse(Float64, A_[8]) : 30.0, :core => length(A_) >= 9 ? parse(Float64, A_[9]) : 30.0,
            :humidity => length(A_) >= 10 ? parse(Float64, A_[10]) : 120.0, :mass => length(A_) >= 11 ? parse(Float64, A_[11]) : 120.0)
NODE_RATIO = length(A_) >= 12 ? parse(Float64, A_[12]) : SHIPPED.node_ratio
LAG_MAX = length(A_) >= 13 ? parse(Int, A_[13]) : max(LAG, SHIPPED.lag_max)   # the window may grow to this while the latents still move (settle test)
PRIOR_RELIN = length(A_) >= 14 ? parse(Float64, A_[14]) : SHIPPED.prior_relin # rebuild the marginal prior when the latents moved this many sds (Inf never)
SLICE_G_REFRESH = length(A_) >= 15 ? parse(Bool, A_[15]) : SHIPPED.slice_g_refresh  # retake slice sensitivities at the relin tolerances (false: frozen at first sight)
JITTER = 0.25                                                    # each interval drawn in step × (1 ± 0.25)

REPO = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "stream_fixture.jl"))
include(joinpath(@__DIR__, "stream_schedule.jl"))
START_MODE = Symbol(get(ENV, "STREAM_START", "legacy"))
NOISE_MODE = Symbol(get(ENV, "STREAM_NOISE", START_MODE === :lighting ? "matched" : "stress"))
NOISE_MODE in (:matched, :stress) || throw(ArgumentError("STREAM_NOISE is matched or stress"))
INITIAL_MODE = Symbol(get(ENV, "STREAM_INITIAL", START_MODE === :lighting ? "prior" : "fixed"))
INITIAL_MODE in (:prior, :fixed) || throw(ArgumentError("STREAM_INITIAL is prior or fixed"))
fixture = stream_fixture(REPO; start=START_MODE)
t, u, pit, cfg = fixture.t, fixture.u, fixture.pit, fixture.cfg
PIECE = :brisket
sys = CookSystem(Slab1D(PIECE; cfg); pit)
blk = system_latents(sys, fixture.sites, fixture.names; probes=[PIECE])
sites, allnames = blk.sites, blk.names
function draw_stream_truth(rng)
    START_MODE === :legacy && return draw_truth(rng, sites)
    # The lighting design's score coordinates are exactly standard normal;
    # draw there directly instead of reusing a discretized beta sampler.
    return [isscore(s) ? randn(rng) : s.p1 + s.p2 * randn(rng) for s in sites]
end
idx = Dict(nm => i for (i, nm) in enumerate(allnames))
AREA, DEPTH = latent_index(blk, "A_meat"), latent_index(blk, "core_depth_fraction")
events = fixture.events
cook_events = [CookEvent(:wrap, PIECE, events.t_wrap), CookEvent(:pull, PIECE, events.t_pull), CookEvent(:add, :pit, events.t_add),
               CookEvent(:spritz, PIECE, events.t_spritz; dt=events.dt_spritz, m=events.m_spritz)]
START_MODE === :lighting && push!(cook_events, CookEvent(:load, PIECE, events.t_load))
t_end = 3600.0 * hours
edges = unique(vcat(3600.0 .* collect(2:2:floor(Int, hours)), [t_end]))
nc = RealCookNoise(; pit_add_bump=NOISE_MODE === :matched ? 0.0 : 2.0)
WHITE = Dict(:pit => 0.1, :core => 0.1, :humidity => 10.0, :mass => 0.005)
OU = Dict(:pit => (nc.pit_sd, nc.pit_tau), :core => (nc.core_sd, nc.core_tau), :humidity => (nc.hum_sd, nc.hum_tau), :mass => (nc.mass_sd, nc.mass_tau))
SENSOR_IDS = (:pit, :core, :humidity, :mass)
observable_of = Dict(:pit => PitTemperature(), :core => TemperatureAt(PIECE; depth_latent=:core_depth_fraction), :humidity => ChamberVapor(), :mass => TotalMass())
sensors = [SensorSpec(id, observable_of[id], NoiseModel(; white_sd=WHITE[id], ar_sd=OU[id][1], tau=OU[id][2], cap=(id === :humidity ? nc.hum_cap : Inf))) for id in SENSOR_IDS]
# STREAM_SN_SCALE scales every process-noise RATE (not the initial widths, not the events' disturbances): the bench's plant
# has no process noise of its own. A small value tests numerical model error;
# it does not make the stochastic state model generatively matched.
const SN_SCALE = parse(Float64, get(ENV, "STREAM_SN_SCALE", "1.0"))
const BENCH_ABSTOL = parse(Float64, get(ENV, "STREAM_ABSTOL", string(SHIPPED.abstol)))
const BENCH_RELTOL = parse(Float64, get(ENV, "STREAM_RELTOL", string(SHIPPED.reltol)))
const BENCH_FLOOR_ABSTOL = parse(Float64, get(ENV, "STREAM_FLOOR_ABSTOL", string(SHIPPED.floor_abstol)))
const BENCH_FLOOR_RELTOL = parse(Float64, get(ENV, "STREAM_FLOOR_RELTOL", string(SHIPPED.floor_reltol)))
all(x -> isfinite(x) && x > 0, (SN_SCALE, BENCH_ABSTOL, BENCH_RELTOL, BENCH_FLOOR_ABSTOL, BENCH_FLOOR_RELTOL)) ||
    throw(ArgumentError("process-rate scale and integration tolerances must be positive and finite"))
const BATCH_S = parse(Float64, get(ENV, "STREAM_BATCH_S", "3600.0"))
const MAX_SLICE_S = parse(Float64, get(ENV, "STREAM_MAX_SLICE_S", string(SHIPPED.max_slice_s)))
const FAIL_FAST = parse(Bool, get(ENV, "STREAM_FAIL_FAST", START_MODE === :lighting ? "true" : "false"))
const VERBOSE_UPDATE = parse(Int, get(ENV, "STREAM_VERBOSE_UPDATE", "0")) # negative: every update
const CHECKPOINT_PATH = get(ENV, "STREAM_CHECKPOINT_PATH", "")
# STREAM_CONDITIONAL="a0,k_spread" runs the conditional (quadrature) backend over those latents instead of the
# Gaussian smoother; STREAM_CONDITIONAL_ORDERS its two nested rules. Its innovation statistics are empty (a
# mixture predictive is not Gaussian): use this mode for the latents, temperatures and cost, not NIS.
const CONDITIONAL_AXES = [Symbol(strip(a)) for a in split(get(ENV, "STREAM_CONDITIONAL", ""), ",") if !isempty(strip(a))]
const CONDITIONAL_ORDERS = Tuple(parse.(Int, split(get(ENV, "STREAM_CONDITIONAL_ORDERS", "3,5"), ",")))
# The JSON-safe part of a conditional update's diagnostics: the coarse-vs-fine rule resolution, or for a failed
# component solve which node (its quadrature coordinates) failed and how far its solve got.
function conditional_diagnostics(d)
    d isa NamedTuple || return nothing
    out = Dict{String,Any}()
    for k in (:mean_error, :covariance_error, :resolved, :order, :node, :coordinates)
        haskey(d, k) && (out[string(k)] = getfield(d, k))
    end
    if haskey(d, :solve)
        s = d.solve
        for k in (:converged, :gradient, :newton_sd, :F)
            haskey(s, k) && (out["solve_" * string(k)] = getfield(s, k))
        end
        out["solve_iters"] = length(get(s, :hist, []))
    end
    return out
end
isfinite(BATCH_S) && BATCH_S > 0 || throw(ArgumentError("STREAM_BATCH_S must be positive and finite"))
sn = let d = StateNoise()
    rates = (:T, :W, :C, :F, :S, :bark, :T_pit, :m_fuel, :m_pan, :T_pan, :T_fb, :ignited)
    StateNoise(; (f => (f in rates ? SN_SCALE * getfield(d, f) : getfield(d, f)) for f in fieldnames(StateNoise))...)
end
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
            meat_present = !(id in (:core, :mass)) || τ >= events.t_load
            loading = events.t_load <= τ < events.t_load + pit.τ_open_load
            τ > 0 && meat_present && !loading && !lid_open_at(τ, events, nc) && push!(ts, τ)
            τ += round(STEP[id] * (1 + JITTER * (2rand(rng) - 1)))
        end
        ts
    end for id in SENSOR_IDS)
end
"The plant in segments saving at every time in `tsave` (the damper on its own grid): the monolith's outputs at those times."
function plant(xs, seg_edges, tsave, A; initial_ics=(;))
    bounds = vcat(1, collect(Int, seg_edges), length(t))
    outs = NamedTuple[]; ics = initial_ics
    for i in 1:(length(bounds) - 1)
        ia, ib = bounds[i], bounds[i + 1]
        tt = t[ia:ib]; uu = u[ia:ib]
        tt[1] >= t_end && break
        stop = min(tt[end], t_end)
        ts = sort(unique(vcat(tt[1], [τ for τ in tsave if tt[1] < τ < stop], stop)))
        φ = xs[i][9:15]
        if i > 1
            ics, a0 = handoff(outs[end], size(outs[end].T_field, 2)); φ = vcat(φ[1:6], a0)
        end
        push!(outs, monolith_solve(xs[i][1:8], φ, uu, tt, ts; pit, cfg, A_meat=A, events..., ics..., abstol=BENCH_ABSTOL, reltol=BENCH_RELTOL))
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
    pit_ics = (Tpit0=ics.Tpit0, mfuel0=ics.mfuel0, mpan0=ics.mpan0, Tpan0=ics.Tpan0, Tfb0=ics.Tfb0, ignited=a)
    x_sys = initial_state(sys, p_pack, play_pack; ics=[piece_ics], pit_ics)
    xb = belief_from_system(x_sys, bl)
    return xb, belief_encode(xb, bl)
end
"The selected instrument model, shared by the bench and its truth diagnostics."
function stream_clean_at(o, k; A, L, depth, cfg=cfg)
    clean = clean_at(o, k; A, L, depth, cfg)
    NOISE_MODE === :matched || return clean
    return merge(clean, (core=E._slab_temperature_at(view(o.T_field, :, k), 1 - depth / L),))
end
"Draw the recipe's initial-state uncertainty from the same coordinate prior as the estimator."
function draw_initial_ics(rng)
    INITIAL_MODE === :fixed && return (;)
    model = SystemModel(sys, CookLog(t, u; events=cook_events, sensors), blk; sn)
    z = [z_moments(s)[1] for s in sites]
    p, play, _ = E._point_params(model, z, 0.0)
    y = belief_encode(belief_from_system(initial_state(sys, p, play), bl), bl)
    Q = belief_cov_to_coord(belief_init_cov(bl, sn), y, bl)
    yt = y + cholesky(Symmetric(E._stable_cov(Q))).L * randn(rng, length(y))
    h = belief_handoff(belief_decode(yt, bl), bl)
    # Ignition comes from the independently drawn a0 latent on the first
    # propagation, exactly as first_slice does in the estimator.
    return merge(h.pieces[1], h.pit)
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

cooks = Any[]; skipped = Int[]; failures = Any[]
active_cook = Ref{Any}(nothing)
const BENCH_BACKEND = backend_fingerprint()
const BENCH_SOURCE = bytes2hex(E.SHA.sha1(vcat(read(@__FILE__), read(joinpath(@__DIR__, "stream_fixture.jl")), read(joinpath(@__DIR__, "stream_schedule.jl")))))
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    write(io, json_document(Dict("plant" => "monolith (1-D, 40 nodes) in segments, constant parameters; streaming real-cook instruments",
        "backend" => "system", "estimator" => "the fixed-lag smoother, lag $LAG (to $LAG_MAX while the latents move), attach $ATTACH, node_ratio $NODE_RATIO",
        "names" => allnames, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped, "steps_s" => Dict(string(k) => v for (k, v) in STEP),
        "jitter" => JITTER, "pace" => PACE, "update_policy" => UPDATE_POLICY,
        "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
        "timing_basis" => "Complete assimilate! wall time, including any compilation in that call; fixture generation, scoring and checkpoint I/O excluded",
        "groups" => group_names, "lag" => LAG, "backend_fingerprint" => BENCH_BACKEND,
        "bench_fingerprint" => BENCH_SOURCE, "start_mode" => string(START_MODE), "noise_mode" => string(NOISE_MODE),
        "initial_mode" => string(INITIAL_MODE), "batch_s" => BATCH_S, "max_slice_s" => MAX_SLICE_S, "lag_max" => LAG_MAX, "prior_relin" => PRIOR_RELIN,
        "slice_g_refresh" => SLICE_G_REFRESH,
        "conditional_axes" => string.(CONDITIONAL_AXES), "conditional_orders" => collect(CONDITIONAL_ORDERS),
        "abstol" => BENCH_ABSTOL, "reltol" => BENCH_RELTOL,
        "floor_abstol" => BENCH_FLOOR_ABSTOL, "floor_reltol" => BENCH_FLOOR_RELTOL,
        "process_rate_scale" => SN_SCALE, "node_ratio" => NODE_RATIO, "load_h" => events.t_load / 3600, "fail_fast" => FAIL_FAST,
        "failures" => failures, "active_cook" => active_cook[],
        "prior" => [Dict("name" => nm, "family" => s.family, "p1" => s.p1, "p2" => s.p2, "p3" => s.p3, "p4" => s.p4, "coord" => string(s.coord)) for (nm, s) in zip(allnames, sites)])))
end
println("stream bench: ", n_cooks, " cooks from seed ", seed0, ", ", hours, " h, lag ", LAG, ", attach ", ATTACH, ", pace ", PACE,
        ", steps ", join(["$k $(v) s" for (k, v) in STEP], ", "), ", node_ratio ", NODE_RATIO); flush(stdout)

for c in 1:n_cooks
    seed = seed0 + c - 1
    rng = MersenneTwister(2000 + seed)
    z0 = draw_stream_truth(rng)
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
        plant(xs, seg_edges, alltimes, Am; initial_ics=draw_initial_ics(MersenneTwister(19000 + seed)))
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the plant failed: %s\n", c, seed, sprint(showerror, err))
        push!(failures, Dict("seed" => seed, "stage" => "plant", "error" => sprint(showerror, err)))
        push!(skipped, seed); save(); continue
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
            cl = stream_clean_at(o, kof[τ]; A=Am, L=xs[1][4], depth, cfg)
            sd = id === :pit ? pit_sd_at(nc, τ, events.t_add) : sd0
            v = getfield(cl, id) + sd * d + WHITE[id] * randn(rs)
            id === :humidity && (v = min(v, nc.hum_cap))
            push!(recs, MeasurementRecord(id, τ, v))
        end
    end
    sort!(recs; by=r -> r.t)
    clog = CookLog(t, u; events=cook_events, sensors, measurements=recs)
    model = SystemModel(sys, clog, blk; sn, abstol=BENCH_ABSTOL, reltol=BENCH_RELTOL,
                        floor_abstol=BENCH_FLOOR_ABSTOL, floor_reltol=BENCH_FLOOR_RELTOL)
    sm = if isempty(CONDITIONAL_AXES)
        online_init(model; lag=LAG, attach=ATTACH, node_ratio=NODE_RATIO, lag_max=LAG_MAX, prior_relin=PRIOR_RELIN,
                    max_slice_s=MAX_SLICE_S, slice_g_refresh=SLICE_G_REFRESH)
    else
        E.conditional_init(model; axes=CONDITIONAL_AXES, orders=CONDITIONAL_ORDERS, lag=LAG, attach=ATTACH,
                           node_ratio=NODE_RATIO, lag_max=LAG_MAX, prior_relin=PRIOR_RELIN, max_slice_s=MAX_SLICE_S)
    end
    base = sm isa E.ConditionalSmoother ? sm.anchor : sm          # the smoother that owns the graph structure
    temps = Any[]; out = NamedTuple[]; failed = nothing
    input_records = [Dict("sensor"=>string(r.sensor), "t_s"=>r.t, "value"=>r.value) for r in recs]
    input_fingerprint = bytes2hex(E.SHA.sha1(json_document([(string(r.sensor),r.t,r.value) for r in recs])))
    active_cook[] = Dict("seed" => seed, "temps" => temps, "n_readings" => length(recs), "n_updates" => 0,
                         "input_records" => input_records, "input_fingerprint" => input_fingerprint)
    save()
    i = 1; clock = 0.0
    while i <= length(recs)
        # the batch: this reading, and under compute pacing every reading that arrived while the last update ran
        j = i
        if PACE < 0
            # Offline diagnostic: look ahead to the next node before updating.
            # The causal policies below never wait to assemble a node's readings.
            k = i
            next_edge = edges[something(findfirst(>=(recs[i].t), edges), length(edges))]
            horizon = min(recs[i].t + BATCH_S, next_edge)
            while k < length(recs) && recs[k + 1].t <= horizon; k += 1; end
            j = i + length(first(E._slice_chunks(base, recs[i:k]))) - 1
        else
            j, horizon = stream_arrival_batch(recs, i, clock; paced=PACE > 0)
        end
        batch = recs[i:j]
        processing_start = PACE > 0 ? horizon : recs[j].t
        # Optional diagnostic restart before the arrival, including the exact
        # readings. This does not enter the measured assimilation wall time.
        isempty(CHECKPOINT_PATH) || serialize(CHECKPOINT_PATH, (sm=sm, recs=batch, seed=seed))
        empty!(E.SLICE_TIMERS); nodes_before = base.node_count
        assimilation_start = time()
        r = try
            assimilate!(sm, batch; verbose=(VERBOSE_UPDATE < 0 || length(out) + 1 == VERBOSE_UPDATE))
        catch err
            err isa InterruptException && rethrow(); failed = err; break
        end
        # A queued batch starts when the previous solve finishes, even when its
        # newest reading is older. Using that reading's time rewound the clock
        # and understated backlog latency.
        assimilation_wall = time() - assimilation_start
        processing_finish = processing_start + (PACE > 0 ? PACE * assimilation_wall : 0.0)
        PACE > 0 && (clock = processing_finish)
        # Score the belief at ITS OWN time: `t_belief` is the record's propagation target (the arrival in an
        # open slice, the newest node otherwise — on an uncommitted update the pre-arrival node). Scoring at
        # the arrival time mislabelled delayed and uncommitted updates by up to a slice. An event node's time
        # is not a reading time; it takes the nearest truth sample.
        tb = get(r, :t_belief, r.t)
        k = haskey(kof, tb) ? kof[tb] : argmin(abs.(o.t .- tb))
        zt = truth_at(z0, dd, tb, events.t_add); ez = r.z .- zt
        # every reading's standardized innovation per sensor (the full batch, not the last per sensor)
        innov_lists = Dict{String,Vector{Float64}}()
        for rr in get(r, :innov_rows, NamedTuple[])
            push!(get!(innov_lists, string(findfirst(==(rr.sensor), SENSOR_IDS)), Float64[]), rr.std)
        end
        _, y_true = truth_belief(o, k)
        eT = [r.T_pit[1] - o.Tpit[k], r.pieces[PIECE].T_centre[1] - o.Tcore[k], r.pieces[PIECE].T_surface[1] - o.Tsurf[k]]
        push!(temps, Dict("t_h" => r.t / 3600, "t_belief_h" => tb / 3600, "k" => k, "n_readings" => length(batch), "committed" => r.committed, "updated" => r.committed,
            "arrivals" => [Dict("sensor" => string(rec.sensor), "t_s" => rec.t) for rec in batch],
            "processing_start_s" => processing_start, "processing_finish_s" => processing_finish,
            "oldest_reading_latency_s" => processing_finish - first(batch).t,
            "newest_reading_latency_s" => processing_finish - last(batch).t,
            "nodes_added" => base.node_count - nodes_before,
            "n_censored" => count(rec -> E.censored_at(sensor(model.log, rec.sensor), rec), batch),
            "n_retry_queued" => length(base.retry), "n_dropped_total" => base.n_dropped,
            "z" => copy(r.z), "z_sd" => sqrt.(max.(diag(r.z_cov), 0.0)), "z_true" => zt,
            "nees_g" => group_nees(r.y .- y_true, r.y_cov),
            "pit" => [r.T_pit[1], r.T_pit[2], o.Tpit[k]], "centre" => [r.pieces[PIECE].T_centre[1], r.pieces[PIECE].T_centre[2], o.Tcore[k]],
            "surface" => [r.pieces[PIECE].T_surface[1], r.pieces[PIECE].T_surface[2], o.Tsurf[k]],
            "nis" => r.nis, "dof" => r.dof, "innov" => innov_lists,
            "innovation_whitened" => [Dict("t" => w.t, "sensor" => string(w.sensor), "value" => w.value) for w in r.innovation_whitened],
            "nees_z" => nees(ez, r.z_cov), "nees_z16" => nees(ez[keep16], r.z_cov[keep16, keep16]), "nees_T" => nees(eT, r.x_cov[[ip, ic, is], [ip, ic, is]]),
            "wall_s" => assimilation_wall, "iters" => r.iters, "attempts" => r.attempts, "gradient" => r.gradient,
            "chunks" => get(r, :chunks, 1), "solver_metrics_scope" => "last internal chunk",
            "newton_sd" => r.newton_sd, "newton_tol" => r.newton_tol,
            "newton_joint_sd" => r.newton_joint_sd, "joint_tol" => r.joint_tol,
            "converged" => r.converged, "n_nodes" => r.n_nodes,
            # why an update did not commit, and the quadrature's own resolution when the backend is conditional
            "reason" => string(get(r, :reason, "")), "diagnostics" => conditional_diagnostics(get(r, :diagnostics, nothing))))
        push!(out, (t=r.t, z=copy(r.z), z_cov=copy(r.z_cov), committed=r.committed))
        active_cook[]["n_updates"] = length(out)
        active_cook[]["wall_filter_s"] = time() - t0w
        active_cook[]["last_z_error_sd"] = ez ./ sqrt.(max.(diag(r.z_cov), 1e-300))
        save()
        if true
            zs_ = ez ./ sqrt.(max.(diag(r.z_cov), 1e-300)); iw = argmax(abs.(zs_))
            @printf("  seed %d t %.2f h: %d updates, %d nodes in window (+%d), last update %.1f s (%d readings, %d chunks; last chunk %d iters, %d attempts, committed %s), NIS/dof %.2f, worst latent %s %+.2f sd, k_spread %+.2f, a0 %+.2f, prior rebuilt %d | %s\n",
                    seed, r.t / 3600, length(out), r.n_nodes, base.node_count - nodes_before, assimilation_wall, length(batch), get(r, :chunks, 1), r.iters, get(r, :attempts, 0), string(r.committed), r.nis / max(r.dof, 1),
                    allnames[iw], zs_[iw], zs_[idx["k_spread"]], zs_[ia0], base.n_prior_relin,
                    join(["$k $(round(v; digits=1))" for (k, v) in sort(collect(E.SLICE_TIMERS); by=last, rev=true)], ", ")); flush(stdout)
        end
        if FAIL_FAST && !r.committed
            # A discarded arrival already fails the complete-cook gate. Keep its
            # diagnostics instead of spending hours continuing an invalid run.
            failed = ErrorException("uncommitted update at $(r.t / 3600) h: gradient=$(r.gradient), newton_sd=$(r.newton_sd)")
            break
        end
        i = j + 1
    end
    if failed !== nothing
        @printf("cook %d (seed %d): the estimator failed at %.2f h (%s); skipped\n", c, seed, isempty(out) ? 0.0 : out[end].t / 3600, sprint(showerror, failed)[1:min(end, 160)])
        push!(failures, Dict("seed" => seed, "stage" => "estimator", "error" => sprint(showerror, failed), "temps" => temps,
                             "input_records" => input_records, "input_fingerprint" => input_fingerprint))
        push!(skipped, seed); active_cook[] = nothing; save(); continue
    end
    available_edges = [(te, stream_available_at(temps,te)) for te in edges]
    batches = [(x = out[j]; row = temps[j]; zt = truth_at(z0, dd, x.t, events.t_add); sd = sqrt.(diag(x.z_cov));
                Dict("t_h" => te / 3600, "state_t_h" => x.t / 3600,
                     "nees_z" => row["nees_z"], "nees_z16" => row["nees_z16"], "nees_T" => row["nees_T"], "nees_g" => row["nees_g"],
                     "pit" => row["pit"], "centre" => row["centre"], "surface" => row["surface"],
                     "z" => x.z, "sd" => sd, "err_sd" => (x.z .- zt) ./ sd, "z_true" => zt,
                     "err_sd_graphfilter" => (x.z .- zt) ./ sd, "err_sd_system" => (x.z .- zt) ./ sd)) for (te,j) in available_edges if j !== nothing]
    push!(cooks, Dict("seed" => seed, "batches" => batches, "temps" => temps, "wall_filter_s" => time() - t0w, "n_updates" => length(out),
                      "n_readings" => length(recs), "n_uncommitted" => count(x -> !x.committed, out),
                      "input_records" => input_records, "input_fingerprint" => input_fingerprint,
                      "unavailable_edges_h" => [te/3600 for (te,j) in available_edges if j === nothing]))
    # The belief statistics are HOLD-TIME weighted: each update's belief counts for the cook time it stood
    # before the next one. A per-update average compares runs whose batching differs on different
    # stretches of the cook (LR2 2026-10-11: a backlogged arm's 354 updates were 60% pre-load and its
    # per-update means looked 4x better than the matched-time truth). NIS stays per update: it scores
    # the innovations each update consumed, not a belief held over time.
    tb = [get(x, "t_belief_h", x["t_h"]) for x in temps]
    hold = length(tb) > 1 ? vcat(max.(diff(tb), 0.0), [0.0]) : [1.0]
    sum(hold) > 0 || (hold = ones(length(tb)))
    wmean(v, w) = (k = isfinite.(v) .& (w .> 0); any(k) ? sum(v[k] .* w[k]) / sum(w[k]) : NaN)
    function wmed(v, w)
        k = isfinite.(v) .& (w .> 0); any(k) || return NaN
        a = abs.(v[k]); p = sortperm(a); cw = cumsum(w[k][p])
        return a[p][searchsortedfirst(cw, cw[end] / 2)]
    end
    zs(key) = [(x[key][1] - x[key][3]) / x[key][2] for x in temps]
    nis_r = Float64[x["nis"] / x["dof"] for x in temps if x["dof"] > 0 && isfinite(x["nis"])]
    @printf("cook %d (seed %d): %d readings in %d updates (%d not committed), %.0f s; time-weighted |err|/sd median pit %.2f centre %.2f surface %.2f; NIS/dof mean %.2f; time-weighted NEES/16 %.2f, NEES_T/3 %.2f\n",
            c, seed, length(recs), length(out), count(x -> !x.committed, out), time() - t0w,
            wmed(zs("pit"), hold), wmed(zs("centre"), hold), wmed(zs("surface"), hold), sum(nis_r) / max(length(nis_r), 1),
            wmean([x["nees_z16"] for x in temps], hold) / 16, wmean([x["nees_T"] for x in temps], hold) / 3)
    gm(g) = wmean([Float64(x["nees_g"][g]) for x in temps], hold)
    println("    the state by group, time-weighted NEES/dof: ", join([@sprintf("%s %.2f", g, gm(g)) for g in group_names], ", "))
    println("    the latents at the edges, err/sd: ", join([@sprintf("%.0fh max %.2f (%s)", b["t_h"], maximum(abs.(b["err_sd"])), allnames[argmax(abs.(b["err_sd"]))]) for b in batches], "; "))
    active_cook[] = nothing
    flush(stdout); save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
# A complete JSON failure record is useful, but a worker with rejected cooks
# must not look successful to the round driver or a shell pipeline.
(isempty(skipped) && all(c -> c["n_uncommitted"] == 0, cooks)) || exit(1)
