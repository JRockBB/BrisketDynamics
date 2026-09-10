# the log-posterior along the widest Gaussian directions at a fit: is it flat, and does the prior act?
include(joinpath(@__DIR__, "particle_probe_setup.jl"))
r = recs[end]
z, V = r.z, Matrix(Symmetric(r.cov))
channels = r.channels
S0, J0, lay = physics_outputs(z, d, ph)
_, _, w = assemble(channels, z, S0, J0, prior, lay); ok = .!isnan.(w)
function logpost(zz)
    S = try physics_stack(zz, d, ph)[1] catch err; err isa InterruptException ? rethrow() : nothing end
    S === nothing && return (-Inf, NaN, NaN)
    a = assemble(channels, zz, S, J0, prior, lay)[1]
    ll = -0.5 * sum(abs2, (w .- a)[ok]); lpr = -prior_terms(zz, prior)[1]
    return (ll + lpr, ll, lpr)
end
F = eigen(Symmetric(V)); idx = sortperm(F.values; rev=true)[1:5]
lp0 = logpost(z)
@printf("at %.0f h: log-posterior at the MAP %.2f (loglik %.2f, log-prior %.2f); %d whitened rows\n", r.t / 3600, lp0..., count(ok))
for (j, i) in enumerate(idx)
    v = F.vectors[:, i]; sdv = sqrt(F.values[i])
    top = sortperm(abs.(v); rev=true)[1:3]
    @printf("direction %d: sd %.3f, mostly %s\n", j, sdv, join(["$(allnames[t]) $(round(v[t]; digits=2))" for t in top], ", "))
    for s in (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
        lp, ll, lpr = logpost(z .+ s * sdv .* v)
        @printf("    s %+5.1f sd: log-post %+9.2f  (loglik %+9.2f, log-prior %+8.2f)  Gaussian would be %+8.2f\n", s, lp - lp0[1], ll - lp0[2], lpr - lp0[3], -0.5 * s^2)
    end
end
