# Replay one control cook to a given arrival and assimilate that arrival verbosely.
include(joinpath(@__DIR__, "..", "..", "estimator", "scripts", "consistency_control.jl"))
seed = parse(Int, ARGS[1]); suite = Symbol(ARGS[2]); t_stop = parse(Float64, ARGS[3])
model = consistency_model(; nx=4, suite, stop=t_stop)
times = collect(0.0:120.0:t_stop)
rng = MersenneTwister(seed)
sm = online_init(model; lag=typemax(Int), record_limit=0)
zt = map(model.blk.sites) do site
    isscore(site) ? randn(rng) : E._betalike(site) ? (b = E.beta_inc_inv(site.p1, site.p2, rand(rng)); log(b[1] / b[2])) : site.p1 + site.p2 * randn(rng)
end
ky = sm.nodes[1][2]
prior = only(f for f in factors_of(sm.graph, ky) if f isa GaussianPrior)
yt = prior.m + prior.R \ randn(rng, length(prior.m))
disc = zeros(length(model.log.sensors))
hs = [compile_observable(s.observable, model.sys, model.bl, model.blk, model.log) for s in model.log.sensors]
for (j, t) in enumerate(times)
    global ky, yt
    if j > 1
        kp = ky; ky = E._node_at!(sm, t); f = sm.dyn[(kp, ky)]
        yt, _ = E.propagate(model, yt, zt, times[j-1], t; first_slice=f.first)
        yt += truth_process_root(model, yt, times[j-1], t, f.events) * randn(rng, length(yt))
    end
    p, _, tk = E._point_params(model, zt, t)
    xs = system_from_belief(belief_decode(yt, model.bl), model.bl)
    readings = MeasurementRecord[]
    for (i, s) in enumerate(model.log.sensors)
        rho, sd = j == 1 ? (0.0, s.noise.ar_sd) : E.ou_step(s.noise, t - times[j-1])
        disc[i] = rho * disc[i] + sd * randn(rng)
        push!(readings, MeasurementRecord(s.id, t, min(hs[i](xs, zt, p, tk, t) + disc[i] + s.noise.white_sd * randn(rng), s.noise.cap)))
    end
    r = assimilate!(sm, readings; verbose=(t == t_stop))
    println("t=$t committed=$(r.committed) iters=$(get(r, :iters, -1)) newton_sd=$(get(r, :newton_sd, NaN))")
    if t == t_stop
        println("latent error / sd at this arrival's truth: ", round.((sm.graph.values[sm.key_z] .- zt) ./ sqrt.(diag(joint_cov(sm.graph; keys=[sm.key_z])[1])); digits=2))
    end
end
