# The smoother on one matched static cook of the bench for its first hours, at lag 0: per arrival the belief of the
# ignited fraction and the fuel mass against the plant's, the latents k_spread, a0, r_max, C_th, D_eff with their widths
# against the truth, the temperatures' NEES, NIS, and the Newton-metric step at the accepted point.
const SEED = parse(Int, get(ENV, "SEED", "1"))
const HOURS = parse(Float64, get(ENV, "HOURS", "4"))
const LAG_ = parse(Int, get(ENV, "LAG", "0"))
empty!(ARGS); append!(ARGS, ["prior_next_cook.json", "1", string(SEED), "run_seed_unused.json", string(HOURS), "0", "default", string(LAG_), "0", "0", "matched"])
include(joinpath(@__DIR__, "worker_setup.jl"))
using Statistics
iign = first(E.belief_index(bl, :pit, :ignited)); imf = first(E.belief_index(bl, :pit, :m_fuel))
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
arrivals = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
clog = CookLog(t, u; events=cook_events, sensors, measurements=recs)
model = SystemModel(sys, clog, blk; sn)
sm = online_init(model; lag=LAG_, als=nothing, drift=nothing)
zt = truth_at(z0, dd, 0.0, events.t_add)
kk(te) = searchsortedlast(t, te + 1e-9)
show_sites = ["k_spread", "a0", "r_max", "C_th", "UA", "D_eff", "h_c"]
println("seed ", seed, " lag ", LAG_, ": truth ", join(["$s $(round(zt[idx[s]]; digits=2))" for s in show_sites], ", "))
@printf("%6s %5s | %-26s | %-22s | %-8s | %s | %5s %5s %6s %5s\n", "t[min]", "wall", "ignited belief (sd) | plant", "m_fuel belief (sd) | plant", "NEES_T", join([@sprintf("%-14s", s) for s in show_sites], ""), "NIS", "grad", "Newton", "it")
println("  latents as (estimate - truth)/sd, the sd after")
t_run = time()
nT = Float64[]; zs = []
for grp in arrivals
    r = assimilate!(sm, grp)
    if !r.committed
        yign = variable_value(sm.graph, sm.nodes[end][2])[iign]
        println("  ", round(r.t / 60; digits=1), " min: not committed; ", r.iters, " iterations in ", r.attempts, " attempts; gradient ", round(r.gradient; sigdigits=3),
                "; Newton step ", round(r.newton_sd; sigdigits=3), " sd; nodes ", r.n_nodes, "; the newest node's ignited coordinate ", round(yign; digits=3), " (a = ", E._belief_dec(yign, :ignited), ")")
        println("    the same arrival again, verbosely (it rolls back again):")
        r2 = assimilate!(sm, grp; verbose=true)
        println("    -> committed ", r2.committed)
        r2.committed || continue
        r = r2
    end
    k = kk(r.t)
    xb_true, y_true = truth_belief(o, k)
    eT = [r.T_pit[1] - o.Tpit[k], r.pieces[PIECE].T_centre[1] - o.Tcore[k], r.pieces[PIECE].T_surface[1] - o.Tsurf[k]]
    neesT = nees(eT, r.x_cov[[ip, ic, is], [ip, ic, is]])
    push!(nT, neesT)
    sd = sqrt.(diag(r.z_cov))
    @printf("%6.1f %5.1f | %.5f (%.1e) | %.5f | %.4f (%.4f) | %.4f | %8.2f | %s | %5.2f %5.0e %6.2f %2d\n", r.t / 60, r.wall,
            r.x[iign], sqrt(r.x_cov[iign, iign]), o.ignited[k], r.x[imf], sqrt(r.x_cov[imf, imf]), o.m_fuel[k], neesT,
            join([@sprintf("%+5.1f (%.3f) ", (r.z[idx[s]] - zt[idx[s]]) / sd[idx[s]], sd[idx[s]]) for s in show_sites], ""), r.nis / max(r.dof, 1), r.gradient, r.newton_sd, r.iters)
    flush(stdout)
end
@printf("done: %d arrivals in %.0f s; NEES_T mean %.2f (n %d)\n", length(arrivals), time() - t_run, mean(nT), length(nT))
