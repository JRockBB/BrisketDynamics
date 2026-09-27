# The collagen field's whitened dynamics residual at the TRUTH over the first two hours of a matched cook, per ten
# minutes, beside how many of the plant's nodes sit inside the encode's floor band (C in (δ, 2δ), δ = 1e-3) and below it:
# where the soft floor's slope differs from the logit's, the encoded plant field and the model's propagation of it
# disagree, and the field's shape whitening (the node jitter) turns that into tens of sds.
const SEED = parse(Int, get(ENV, "SEED", "1"))
empty!(ARGS); append!(ARGS, ["prior_next_cook.json", "1", "1", "c_band_unused.json", "3", "0", "default", "0", "0", "0", "matched"])
include(joinpath(@__DIR__, "worker_setup.jl"))
using Statistics
rC = E.belief_index(bl, PIECE, :C); rT = E.belief_index(bl, PIECE, :T)
groups = [(g, rg) for (g, rg) in bl.groups]
seed = SEED
rng = MersenneTwister(2000 + seed)
z0 = draw_truth(rng, sites)
dd0 = DriftDraw(rng, idx; sites)
dd = DriftDraw(Dict(k => zero(v) for (k, v) in dd0.inc), Dict(k => 0.0 for (k, v) in dd0.jump))
seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
xs = [from_z(z, sites) for z in blocks]
A, depth_fraction = xs[1][AREA], xs[1][DEPTH]
depth = probe_depth(xs[1][4], depth_fraction)
o = solve_segments([x[1:8] for x in xs], [x[9:15] for x in xs], u, t, seg_edges; pit, cfg, A_meat=A, events...)
readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=xs[1][4], depth=depth, cfg)
k_end = searchsortedlast(t, t_end + 1e-9)
recs = MeasurementRecord[]
for k in 1:k_end, id in SENSOR_IDS
    getfield(masks, id)[k] && push!(recs, MeasurementRecord(id, t[k], getfield(readings, id)[k]))
end
clog = CookLog(t, u; events=cook_events, sensors, measurements=recs)
model = SystemModel(sys, clog, blk; sn)
zt = truth_at(z0, dd, 0.0, events.t_add)
kz = VarKey(:cook, :z, 0)
δ = E.BELIEF_ENC_MIN_UNIT
println("seed ", seed, ": the collagen field's ", length(rC), " nodes; floor δ = ", δ)
@printf("%-9s %-12s %-12s %-12s | plant C: %-8s %-8s %-8s | %s\n", "t [min]", "C whitened", "T whitened", "worst node", "max", "n<δ", "n in band", "the worst node's C (plant, model's propagation) and its whitened residual")
for k in 1:5:(k_end - 1)
    t[k] > 2.2 * 3600 && break
    ka, kb = VarKey(:cook, :y, k), VarKey(:cook, :y, k + 1)
    _, y0 = truth_belief(o, k); _, y1 = truth_belief(o, k + 1)
    f = DynamicsFactor(model, kz, ka, kb, t[k], t[k + 1]; first=(k == 1), events=sensor_events(clog, t[k], t[k + 1]))
    r = E.residual(f, [zt, y0, y1])
    Cp = o.C_field[:, k + 1]
    x1 = belief_decode(y1, bl); xf = belief_decode(f.cache.f, bl)
    j = argmax(abs.(r[rC]))
    @printf("%7.1f   %10.2f   %10.2f   node %-6d | %8.2e %8d %8d | %.3e %.3e  %+8.2f\n", t[k + 1] / 60, sqrt(mean(abs2, r[rC])), sqrt(mean(abs2, r[rT])), j,
            maximum(Cp), count(<(δ), Cp), count(c -> δ <= c < 2δ, Cp), x1[rC[j]], xf[rC[j]], r[rC[j]])
end
