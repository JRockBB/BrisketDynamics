# Does the estimator's model reproduce the bench's plant at the truth? On one stream_bench cook: from the plant's true state at
# each node time, the estimator's propagation with the TRUE latents to the next node against the plant's true state there
# (whitened by the dynamics factor's own process covariance, per state group; and in physical units for the pit, the fire,
# the fuel and the centre), and each sensor's model prediction at the true state against the plant's clean reading.
# Run: julia --threads=4 --project=estimator estimator/scripts/truth_residuals.jl [seed] [hours] [node_spacing_s]
seed = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
hours_chk = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 2.0
spacing = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 300.0
empty!(ARGS); push!(ARGS, "1", string(seed), "truth_tmp.json", string(hours_chk), "2", "slice", "-1")
src = read(joinpath(@__DIR__, "stream_bench.jl"), String)
include_string(Main, src[1:first(findfirst("for c in 1:n_cooks", src)) - 1], joinpath(@__DIR__, "stream_bench.jl"))

rng = MersenneTwister(2000 + seed)
z0 = draw_truth(rng, sites); dd0 = DriftDraw(rng, idx; sites)
dd = DriftDraw(Dict(k => zero(v) for (k, v) in dd0.inc), Dict(k => 0.0 for (k, v) in dd0.jump))
seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
xs = [from_z(z, sites) for z in blocks]
Am, depth = xs[1][AREA], probe_depth(xs[1][4], xs[1][DEPTH])
node_t = collect(0.0:spacing:3600hours_chk)
o = plant(xs, seg_edges, node_t[2:end], Am)
kof = Dict(τ => k for (k, τ) in enumerate(o.t))
model = SystemModel(sys, CookLog(t, u; events=cook_events, sensors), blk; sn)
zt = truth_at(z0, dd, 0.0, events.t_add)
ib(f) = E.belief_index(bl, f === :T ? PIECE : :pit, f)
println("seed $seed: true a0 $(round(zt[ia0]; digits=2)), k_spread $(round(zt[idx["k_spread"]]; digits=2)) (latent coordinates)")
@printf("%7s | %-62s | %s\n", "slice", "whitened residual per group, mean square per coordinate (1 = the process noise's size)", "physical: pit K, ignited, fuel kg, centre K")
for j in 1:(length(node_t) - 1)
    ta, tb = node_t[j], node_t[j + 1]
    _, ya = truth_belief(o, kof[ta]); xb_true, yb = truth_belief(o, kof[tb])
    yp, _ = E.propagate(model, ya, zt, ta, tb; first_slice=(j == 1))
    Q = belief_process_cov(bl, sn, (tb - ta) / 3600; events=E.sensor_events(model.log, ta, tb))
    f = DynamicsFactor(model, VarKey(:params, :x, 0), VarKey(:state, :x, 1), VarKey(:state, :x, 2), ta, tb; first=(j == 1), events=E.sensor_events(model.log, ta, tb))
    W = E._whitener(f, yp)
    r = W * (yb .- yp)
    gs = join([@sprintf("%s %.2g", g, sum(abs2, r[rg]) / length(rg)) for (g, rg) in zip(group_names, group_ranges) if sum(abs2, r[rg]) / length(rg) > 0.5], " ")
    xp = belief_decode(yp, bl)
    d(f) = xb_true[first(ib(f))] - xp[first(ib(f))]
    @printf("%5.0f m | %-62s | %+.3f %+.2e %+.4f %+.3f\n", tb / 60, isempty(gs) ? "all groups < 0.5" : gs, d(:T_pit), d(:ignited), d(:m_fuel), d(:T))
end
println("observations: the model's prediction at the true state minus the plant's clean reading")
hs = Dict(s.id => compile_observable(s.observable, sys, bl, blk, model.log) for s in sensors)
for τ in node_t[2:end]
    xb_true, _ = truth_belief(o, kof[τ])
    p, _, tk = E._point_params(model, zt, τ)
    xsys = system_from_belief(xb_true, bl)
    cl = clean_at(o, kof[τ]; A=Am, L=xs[1][4], depth, cfg)
    @printf("%5.0f m | %s\n", τ / 60, join([@sprintf("%s %+.3g", id, hs[id](xsys, zt, p, tk, τ) - getfield(cl, id)) for id in SENSOR_IDS], "  "))
end
