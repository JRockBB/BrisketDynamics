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
    r = assimilate!(sm, readings)
    println("t=$t committed=$(r.committed) iters=$(get(r, :iters, -1)) newton_sd=$(get(r, :newton_sd, NaN))")
end

# the profile of the objective along one latent: pinned at the MAP ± k GN-marginal sds, everything else re-solved
idx = parse(Int, ARGS[4])
g = sm.graph; kz = sm.key_z
m = copy(g.values[kz]); C = joint_cov(g; keys=[kz])[1]; sd = sqrt(C[idx, idx])
F0 = E._objective_at(g, E._stacked(g, g.keys, E._offsets(g, g.keys)[1], E._offsets(g, g.keys)[2]), E._offsets(g, g.keys)[1])
println("latent $(model.blk.names[idx]): MAP $(round(m[idx]; digits=3)), GN sd $(round(sd; digits=3)), truth $(round(zt[idx]; digits=3)) ($(round((zt[idx]-m[idx])/sd; digits=2)) sd); F at MAP $(round(F0; digits=5))")
iT = first(E.belief_index(model.bl, :brisket, :T))
ky_last = sm.nodes[end][2]
for k in (-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
    g2 = deepcopy(g)
    pin = copy(m); pin[idx] = m[idx] + k * sd
    R = zeros(1, length(m)); R[1, idx] = 1e4
    f = E.GaussianPrior(kz, pin, R; source="pin")
    add_factor!(g2, f)
    g2.values[kz][idx] = pin[idx]
    rc = gauss_newton!(g2; iters=120, tol=1e-4, ftol=1e-2, step_bounds=Dict(kz => 0.25), expand=4)
    remove_factor!(g2, f)
    off, n = E._offsets(g2, g2.keys)
    F = E._objective_at(g2, E._stacked(g2, g2.keys, off, n), off)
    Tc = belief_decode(g2.values[ky_last], model.bl)[iT]
    println("  k=$(k): ΔF $(round(F - F0; digits=3))  (Gaussian ½k² = $(0.5k^2))  converged $(rc.converged)  centre T $(round(Tc; digits=2))")
    flush(stdout)
end
