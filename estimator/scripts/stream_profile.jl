# The objective's profile along latents at the end of a stream_bench cook's first hours: the cook assimilated (in the
# bench's node-sized batches), then each named latent pinned at the estimate + k reported sds (toward and away from the
# truth), everything else re-solved, the objective's rise against the Gaussian's ½k².
# Run: STREAM_SN_SCALE=0.001 julia --threads=4 --project=estimator estimator/scripts/stream_profile.jl [seed] [hours] [lag] [node_ratio] latent...
seed = parse(Int, ARGS[1]); hours_p = parse(Float64, ARGS[2]); lag_p = ARGS[3]; ratio_p = ARGS[4]; prof = ARGS[5:end]
empty!(ARGS); push!(ARGS, "1", string(seed), "profile_tmp.json", string(hours_p), lag_p, "slice", "-1", "30", "30", "120", "120", ratio_p)
src = read(joinpath(@__DIR__, "stream_bench.jl"), String)
include_string(Main, src[1:first(findfirst("for c in 1:n_cooks", src)) - 1], joinpath(@__DIR__, "stream_bench.jl"))

rng = MersenneTwister(2000 + seed)
z0 = draw_truth(rng, sites); dd0 = DriftDraw(rng, idx; sites)
dd = DriftDraw(Dict(k => zero(v) for (k, v) in dd0.inc), Dict(k => 0.0 for (k, v) in dd0.jump))
seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
xs = [from_z(z, sites) for z in blocks]
Am, depth = xs[1][AREA], probe_depth(xs[1][4], xs[1][DEPTH])
rs = MersenneTwister(9000 + seed); sched = schedules(rs)
o = plant(xs, seg_edges, sort(unique(vcat(values(sched)...))), Am); kof = Dict(τ => k for (k, τ) in enumerate(o.t))
recs = MeasurementRecord[]
for id in SENSOR_IDS
    d = 0.0; tp = -Inf; sd0, τc = OU[id]
    for τ in sched[id]
        ρ = isfinite(tp) ? exp(-(τ - tp) / τc) : 0.0
        d = ρ * d + sqrt(1 - ρ^2) * randn(rs); tp = τ
        cl = clean_at(o, kof[τ]; A=Am, L=xs[1][4], depth, cfg)
        v = getfield(cl, id) + (id === :pit ? pit_sd_at(nc, τ, events.t_add) : sd0) * d + WHITE[id] * randn(rs)
        push!(recs, MeasurementRecord(id, τ, id === :humidity ? min(v, nc.hum_cap) : v))
    end
end
sort!(recs; by=r -> r.t)
model = SystemModel(sys, CookLog(t, u; events=cook_events, sensors, measurements=recs), blk; sn)
sm = online_init(model; lag=LAG, attach=:slice, node_ratio=NODE_RATIO)
i = 1
while i <= length(recs)
    k = i; while k < length(recs) && recs[k + 1].t <= recs[i].t + 3600.0; k += 1; end
    j = i + length(first(E._slice_chunks(sm, recs[i:k]))) - 1
    r = assimilate!(sm, recs[i:j]); r.committed || error("uncommitted at $(r.t)")
    global i = j + 1
end
zt = truth_at(z0, dd, recs[end].t, events.t_add)
g = sm.graph; kz = sm.key_z
m = copy(g.values[kz]); C = joint_cov(g; keys=[kz])[1]
off, n = E._offsets(g, g.keys); F0 = E._objective_at(g, E._stacked(g, g.keys, off, n), off)
for nm in prof
    ix = idx[nm]; sd = sqrt(C[ix, ix]); toward = sign(zt[ix] - m[ix])
    @printf("%s: estimate %.3f, reported sd %.3f, truth %.3f (%+.2f sd); the objective's rise with the latent pinned at estimate + k sd:\n", nm, m[ix], sd, zt[ix], (zt[ix] - m[ix]) / sd)
    for k in toward .* [1.0, 2.0, 3.0, 4.0, -1.0, -2.0]
        g2 = deepcopy(g)
        pin = copy(m); pin[ix] = m[ix] + k * sd
        R = zeros(1, length(m)); R[1, ix] = 1e4
        f = E.GaussianPrior(kz, pin, R; source="pin"); add_factor!(g2, f); g2.values[kz][ix] = pin[ix]
        rc = gauss_newton!(g2; iters=80, tol=1e-4, ftol=1e-2, step_bounds=Dict(kz => 0.25), expand=4)
        remove_factor!(g2, f)
        o2, n2 = E._offsets(g2, g2.keys); F = E._objective_at(g2, E._stacked(g2, g2.keys, o2, n2), o2)
        @printf("   k %+.0f (%s the truth): rise %.3f (Gaussian %.1f), converged %s\n", k, sign(k) == toward ? "toward" : "away from", F - F0, 0.5k^2, rc.converged)
        flush(stdout)
    end
end
