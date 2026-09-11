include(joinpath(@__DIR__, "amyosin_setup.jl"))
seed = parse(Int, ARGS[1]); steps = parse(Int, ARGS[2])
ik = findfirst(==("k_e"), allnames); im = findfirst(==("k_m"), allnames)
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
# the Gaussian at 14 h under these channels (the record has no covariance): one fit, warm-started at the record's MAP
fit = stream_fit(chs, prior, d, [14 * 3600.0]; physics=ph, noise=nothing, levelled=false, verbose=false, c0=z)[end]
z, V, sd = fit.z, Matrix(Symmetric(fit.cov)), fit.sd
t0 = time()
rc = ridge_chain(chs, prior, d, ph, z, V; k=17, steps=steps, burn=steps ÷ 4, rng=MersenneTwister(77 + seed), thin=10)
@printf("seed %d, %d steps in %.0f s: accept %.2f, ESS a_myosin %.0f k_e %.0f k_m %.0f (min over sites %.0f)\n", seed, steps, time() - t0, rc.accept, rc.ess[ia], rc.ess[ik], rc.ess[im], minimum(rc.ess))
for (nm, i) in (("a_myosin", ia), ("k_e", ik), ("k_m", im))
    @printf("  %-9s MAP %+.3f  chain mean %+.3f  truth %+.3f | (MAP-truth)/sd %+.2f  (mean-truth)/sd %+.2f  (mean-MAP)/sd %+.2f | sd MAP %.3f chain %.3f\n",
            nm, z[i], rc.z[i], z_true[i], (z[i] - z_true[i]) / sd[i], (rc.z[i] - z_true[i]) / sd[i], (rc.z[i] - z[i]) / sd[i], sd[i], rc.sd[i])
end
