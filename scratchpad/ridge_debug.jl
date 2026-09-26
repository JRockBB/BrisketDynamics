include(joinpath(@__DIR__, "particle_probe_setup.jl"))
r = recs[end]
rc = ridge_chain(r.channels, prior, d, ph, r.z, r.cov; k=5, steps=120, rng=MersenneTwister(3))
println("accept ", rc.accept, "; ess ", round.(rc.ess; digits=1))
println("ridge sd (Gaussian) ", round.(rc.ridge_sd_gauss; digits=3))
println("ridge sd (chain)    ", round.(sqrt.(diag(rc.ridge_cov)); digits=3))
println("ridge mean shift    ", round.(rc.ridge_mean; digits=3))
println("sd ratio per site   ", round.(rc.sd ./ r.sd; digits=2))
# the draws' log-posteriors
S0, J0, lay = physics_outputs(r.z, d, ph); _, _, w = assemble(r.channels, r.z, S0, J0, prior, lay); ok = .!isnan.(w)
lpf(zz) = (S = physics_stack(zz, d, ph)[1]; a = assemble(r.channels, zz, S, J0, prior, lay)[1]; -0.5 * sum(abs2, (w .- a)[ok]) - prior_terms(zz, prior)[1])
lp0 = lpf(r.z)
println("log-post at the MAP ", round(lp0; digits=2), "; at the kept draws (rel.): ", round.([lpf(zz) - lp0 for zz in rc.draws[1:min(end, 6)]]; digits=2))
println("first draws' ridge coords: ", [round.(rc.directions' * (zz .- r.z); digits=2) for zz in rc.draws[1:min(end, 3)]])
