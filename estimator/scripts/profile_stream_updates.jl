# Where a streaming update's time goes: per update, the slice and dynamics factors' Jacobians and solves, the iterations,
# the wall. Uses stream_bench.jl's setup for one cook's first readings.
push!(ARGS, "1", "1", "profile_tmp.json", "1.1", "2", "slice", "0")
const STOP_AFTER = 14
# run the bench's setup without its cook loop
src = read(joinpath(@__DIR__, "stream_bench.jl"), String)
cut = findfirst("for c in 1:n_cooks", src)
include_string(Main, src[1:first(cut) - 1], joinpath(@__DIR__, "stream_bench.jl"))
seed = 1
rng = MersenneTwister(2000 + seed)
z0 = draw_truth(rng, sites); dd0 = DriftDraw(rng, idx; sites)
dd = DriftDraw(Dict(k => zero(v) for (k, v) in dd0.inc), Dict(k => 0.0 for (k, v) in dd0.jump))
seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
xs = [from_z(z, sites) for z in blocks]
Am, depth = xs[1][AREA], probe_depth(xs[1][4], xs[1][DEPTH])
rs = MersenneTwister(9000 + seed); sched = schedules(rs)
alltimes = sort(unique(vcat(values(sched)...)))
o = plant(xs, seg_edges, alltimes, Am); kof = Dict(τ => k for (k, τ) in enumerate(o.t))
recs = MeasurementRecord[]
for id in SENSOR_IDS, τ in sched[id]
    cl = clean_at(o, kof[τ]; A=Am, L=xs[1][4], depth, cfg)
    push!(recs, MeasurementRecord(id, τ, min(getfield(cl, id), id === :humidity ? nc.hum_cap : Inf)))
end
sort!(recs; by=r -> r.t)
model = SystemModel(sys, CookLog(t, u; events=cook_events, sensors, measurements=recs), blk; sn)
sm = online_init(model; lag=2, attach=:slice, node_ratio=0.15)
counts(sm) = (sj=sum((f.n_jacobians for f in values(sm.slices)); init=0), ss=sum((f.n_solves for f in values(sm.slices)); init=0),
              dj=sum((f.n_jacobians for f in values(sm.dyn)); init=0), ds=sum((f.n_solves for f in values(sm.dyn)); init=0))
# batches of the readings in successive 5-minute spans (a profiling stand-in for compute pacing)
spans = [0.0, 16.0, 625.0]
for (n, t0s) in enumerate(spans[1:end-1])
    batch = [r for r in recs if t0s < r.t <= spans[n + 1]]
    isempty(batch) && continue
    empty!(E.SLICE_TIMERS)
    tw = time()
    res = assimilate!(sm, batch; verbose=true)
    tm = E.SLICE_TIMERS
    @printf("update %d (%d readings, to %.0f s): wall %.1f s, iters %d, nodes %d | %s
", n, length(batch), batch[end].t, time() - tw, res.iters, res.n_nodes,
            join(["$k $(round(v; digits=1))" for (k, v) in sort(collect(tm); by=last, rev=true)], ", "))
    flush(stdout)
end
