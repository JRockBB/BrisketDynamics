# The streaming estimator: at every batch boundary (an hour of the cook),
# a FULL-HISTORY refit of the latents on every reading so far, with the
# original prior — exact Gauss-Newton, not a filter with invented process
# noise — warm-started at the previous posterior mean so it converges in
# a few outer steps. Included by BrisketEstimator.
#
# The graph is the gate's: one latent block, one affine node per whitened
# residual, the local-quadratic prior node; the readings come from the
# channel registry (channels.jl). Static readings (rulers, scales before
# the cook) are present from the first batch.

export estimator_setup, restrict, solve_map_channels, stream_fit, graph_posterior_affine

"""
    estimator_setup(t, u, pit, cfg, prior_phys, names; abstol, reltol) -> GateData

A `GateData` for a cook with no reference fit: the log grid, the damper
schedule, the pit and meat configs and the physics priors are what the
physics needs; the gate-only fields are empty.
"""
function estimator_setup(t, u, pit::PitConfig, cfg::MeatConfig, prior_phys::Vector{PriorSite},
                         names::Vector{String}; abstol=1e-6, reltol=1e-6)
    z0 = [isbeta(p) ? log(p.p1 / p.p2) : p.p1 for p in prior_phys]   # medians / Beta mean in z
    return GateData(collect(Float64, t), collect(Float64, u), zeros(0, 2 * length(t)), Float64[],
                    names, prior_phys, z0, fill(NaN, N_PHYS), fill(NaN, N_PHYS, N_PHYS), NaN,
                    pit, cfg, abstol, reltol)
end

# a channel restricted to the readings at times where `keep` is true
restrict(ch::PitProbe, keep) = PitProbe(ch.σ, ch.ρ, ch.mask .& keep, ch.data)
restrict(ch::SurfaceIR, keep) = SurfaceIR(ch.σ, ch.ρ, ch.mask .& keep, ch.data)
restrict(ch::HumiditySensor, keep) = HumiditySensor(ch.σ, ch.ρ, ch.mask .& keep, ch.data)
restrict(ch::MassScale, keep) = MassScale(ch.σ, ch.ρ, ch.mask .& keep, ch.A_meat, ch.area_latent,
                                          ch.shape_latent, ch.data)
restrict(ch::MeatProbe, keep) = MeatProbe(ch.s_fixed, ch.depth_fixed, ch.depth_latent, ch.σ, ch.ρ,
                                          ch.mask .& keep, ch.data)
restrict(ch::FuelScale, keep) = FuelScale(ch.σ, ch.ρ, ch.mask .& keep, ch.data)
restrict(ch::BarkLightness, keep) = BarkLightness(ch.σ, ch.ρ, ch.mask .& keep, ch.data)
restrict(ch::WetBulb, keep) = WetBulb(ch.σ, ch.ρ, ch.mask .& keep, ch.p_atm, ch.a_psy, ch.data)
restrict(ch::StaticReading, keep) = ch

"""
    graph_posterior_affine(c, a, B, w, prior; bp_iters) -> (mean, cov)

One message-passing solve of the graph linearized at `c` from the affine
node data `(a, B)` and the whitened observations `w`.
"""
function graph_posterior_affine(c, a, B, w, prior; bp_iters::Int=20)
    AFF[] = Affine(a, B, copy(c))
    _, g, h = prior_terms(c, prior)
    mp = c .- g ./ h
    Λp = Matrix(Diagonal(h))
    r = infer(model = cook_graph(mp = mp, Λp = Λp), data = (w = w,), meta = GRAPH_META,
              initialization = init_z(mp, Λp), iterations = bp_iters,
              returnvars = (z = KeepLast(),))
    m, V = mean_cov(r.posteriors[:z])
    return collect(m), Matrix(V)
end

_objective(a, w, z, prior) = 0.5 * sum(abs2, w .- a) + prior_terms(z, prior)[1]

"""
    solve_map_channels(channels, prior, d; c0, area_latent, iters, tol, verbose)

The gate's outer Gauss-Newton loop over a channel registry: every outer
step solves the physics and its stacked Jacobian once, assembles the
whitened readings, and takes the graph's posterior mean as the step,
halved while the objective does not decrease; stops on a small step or
on an objective decrease below `ftol` (the adaptive solve's jitter floor
is ~1e-3 in whitened units, so chasing it buys nothing).
"""
function solve_map_channels(channels, prior, d::GateData; c0, area_latent::Int=0,
                            iters::Int=30, tol::Float64=1e-4, ftol::Float64=1e-3,
                            bp_iters::Int=20, verbose::Bool=false)
    c = copy(c0)
    S, J, lay = physics_outputs(c, d; area_latent)
    a, B, w = assemble(channels, c, S, J, prior, lay)
    any(isnan, w) && throw(ArgumentError("every channel in a fit must carry data"))
    Fc = _objective(a, w, c, prior)
    hist = NamedTuple[]
    m, V = c, zeros(length(c), length(c))
    J0 = zeros(size(J))          # a zero Jacobian: predictions only, in the line search
    for k in 1:iters
        t0 = time()
        m, V = graph_posterior_affine(c, a, B, w, prior; bp_iters)
        step = m - c
        α = 1.0
        c_new = c + step
        Sn, _ = physics_stack(c_new, d; area_latent)
        an, _, _ = assemble(channels, c_new, Sn, J0, prior, lay)
        Fn = _objective(an, w, c_new, prior)
        while Fn > Fc && α > 1e-3
            α /= 2
            c_new = c + α * step
            Sn, _ = physics_stack(c_new, d; area_latent)
            an, _, _ = assemble(channels, c_new, Sn, J0, prior, lay)
            Fn = _objective(an, w, c_new, prior)
        end
        push!(hist, (it=k, F=Fc, step=norm(step), alpha=α, wall=time() - t0))
        verbose && @printf("    it %2d  F=%10.4f -> %10.4f  |step|=%.3g  alpha=%.3g  (%.1f s)\n",
                           k, Fc, Fn, norm(step), α, time() - t0)
        verbose && flush(stdout)
        Fn > Fc && break
        converged = (α * norm(step) < tol) || (Fc - Fn < ftol)   # step or objective floor
        c, Fc = c_new, Fn
        S, J, _ = physics_outputs(c, d; area_latent)     # the Jacobian once per accepted step
        a, B, _ = assemble(channels, c, S, J, prior, lay)
        converged && break
    end
    m, V = graph_posterior_affine(c, a, B, w, prior; bp_iters)
    mis = misfit(channels, a, B, w, V)
    return (z=c, mean=m, cov=V, F=Fc, hist=hist, misfit=mis)
end

"""
    stream_fit(channels, prior, d, batch_edges; area_latent, c0, verbose) -> Vector

Refit at each time in `batch_edges` [s] on the readings up to that time
(static readings always), warm-started at the previous posterior mean.
Returns one record per batch: time, posterior mean, sd, covariance,
objective, outer iterations, wall.
"""
function stream_fit(channels, prior, d::GateData, batch_edges; area_latent::Int=0,
                    c0=nothing, verbose::Bool=true, kw...)
    c = c0 === nothing ? vcat(d.z0, [isbeta(p) ? log(p.p1 / p.p2) : p.p1 for p in prior[(N_PHYS + 1):end]]) :
        copy(c0)
    out = NamedTuple[]
    for (b, te) in enumerate(batch_edges)
        keep = d.t .<= te + 1e-9
        chs = [restrict(ch, keep) for ch in channels]
        t0 = time()
        res = solve_map_channels(chs, prior, d; c0=c, area_latent, verbose=verbose, kw...)
        c = res.z
        sd = sqrt.(diag(res.cov))
        push!(out, (batch=b, t=te, z=res.z, sd=sd, cov=res.cov, F=res.F,
                    iters=length(res.hist), wall=time() - t0, misfit=res.misfit))
        verbose && @printf("  batch %2d  t = %5.1f h  F = %9.3f  outer %2d  (%.0f s)\n",
                           b, te / 3600, res.F, length(res.hist), time() - t0)
        verbose && flush(stdout)
    end
    return out
end
