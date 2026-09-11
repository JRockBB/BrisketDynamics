include(joinpath(@__DIR__, "amyosin_setup.jl"))
ik = findfirst(==("k_e"), allnames); im = findfirst(==("k_m"), allnames)
function channels_at(seed)
    c = recs[seed]; b = c["batches"][end]
    z = Float64.(b["z"]); sd = Float64.(b["sd"]); z_true = Float64.(c["z_true"])
    rng = MersenneTwister(1000 + seed); zt = draw_truth(rng, prior); xt = from_z(zt)
    θ, φ, A, depth = xt[1:8], xt[9:15], xt[16], xt[17]
    o = monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, events...)
    readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=θ[4], depth=depth)
    keep = t .<= 14 * 3600
    chs = [PitProbe(truth_noise["pit"]..., Bool.(masks.pit .& keep); data=Float64.(readings.pit)),
           MeatProbe(truth_noise["core"]..., Bool.(masks.core .& keep); depth_latent=DEPTH, data=Float64.(readings.core)),
           CensoredHumidity(truth_noise["humidity"]..., Bool.(masks.humidity .& keep); cap=nc.hum_cap, data=Float64.(readings.humidity)),
           MassScale(truth_noise["mass"]..., Bool.(masks.mass .& keep); data=Float64.(readings.mass))]
    return chs, z, sd, z_true
end
for seed in (3, 1)
    chs, z, sd, z_true = channels_at(seed)
    S0, J0, lay = physics_outputs(z, d, ph); _, _, w = assemble(chs, z, S0, J0, prior, lay); ok = .!isnan.(w)
    lp(zz) = (S = physics_stack(zz, d, ph)[1]; a = assemble(chs, zz, S, J0, prior, lay)[1]; -0.5 * sum(abs2, (w .- a)[ok]) - prior_terms(zz, prior)[1])
    lp0 = lp(z)
    # the plane (a_myosin, k_e) at +-3 marginal sd, 11 x 11, the rest at the MAP: the conditional-on-the-rest joint
    ss = collect(range(-3.0, 3.0; length=11))
    G = [ (zz = copy(z); zz[ia] += sa * sd[ia]; zz[ik] += sk * sd[ik]; lp(zz) - lp0) for sa in ss, sk in ss ]
    P = exp.(G .- maximum(G)); P ./= sum(P)
    ma = sum(P .* ss); mk = sum(P .* ss')
    @printf("seed %d: MAP a_myosin %.3f k_e %.3f; truth %.3f %.3f (err/sd %+.2f %+.2f)\n", seed, z[ia], z[ik], z_true[ia], z_true[ik], (z[ia] - z_true[ia]) / sd[ia], (z[ik] - z_true[ik]) / sd[ik])
    @printf("  over the (a_myosin, k_e) plane with the rest at the MAP: mean at (%+.2f, %+.2f) sd from the mode; corr %.2f\n", ma, mk, (sum(P .* (ss .- ma) .* (ss' .- mk))) / sqrt(sum(P .* (ss .- ma) .^ 2) * sum(P .* (ss' .- mk) .^ 2)))
    println("  log-post along the diagonal a_myosin up / k_e down (the compensating direction), s = -3..3: ", join([@sprintf("%+.1f", G[i, 12 - i]) for i in 1:11], " "))
    println("  log-post along a_myosin alone (k_e at the MAP):                                   ", join([@sprintf("%+.1f", G[i, 6]) for i in 1:11], " "))
    flush(stdout)
end
