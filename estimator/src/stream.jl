# The batch estimator: at every batch boundary,
# a FULL-HISTORY refit of the latents on every reading so far, with the
# original prior — exact Gauss-Newton, not a filter with invented process
# noise — warm-started at the previous posterior mean so it converges in
# a few outer steps. Included by BrisketEstimator.
#
# The graph is the gate's: one latent block, one affine node per whitened
# residual or local censored-likelihood site, the local-quadratic prior node; the readings come from the
# channel registry (channels.jl). Static readings (rulers, scales before
# the cook) are present from the first batch.

export estimator_setup, solve_map_channels, stream_fit, graph_posterior_affine

"""
    estimator_setup(t, u, pit, cfg, prior_phys, names; abstol, reltol) -> GateData

A `GateData` for a cook with no reference fit: the log grid, the damper
schedule, the pit and meat configs and the physics priors are what the
physics needs; the gate-only fields are empty.
"""
function estimator_setup(t, u, pit::PitConfig, cfg::MeatConfig, prior_phys::Vector{PriorSite},
                         names::Vector{String}; abstol=1e-6, reltol=1e-6)
    z0 = z_center.(prior_phys)   # medians / Beta mean in z
    return GateData(collect(Float64, t), collect(Float64, u), zeros(0, 2 * length(t)), Float64[],
                    names, prior_phys, z0, fill(NaN, N_PHYS), fill(NaN, N_PHYS, N_PHYS), NaN,
                    pit, cfg, abstol, reltol)
end

"""
    graph_posterior_affine(c, a, B, w, prior; bp_iters) -> (mean, cov)

One message-passing solve of the graph linearized at `c` from the affine
node data `(a, B)` and the whitened observations `w`; `damping` adds a
Levenberg-Marquardt penalty to the prior node's precision (the
local-quadratic prior goes flat where a Beta site sits at an extreme
logit, and an undamped step then runs to infinity).
"""
function graph_posterior_affine(c, a, B, w, prior; bp_iters::Int=20, damping::Float64=0.0)
    AFF[] = Affine(a, B, copy(c))
    mp, Λp = prior_quadratic(c, prior; damping)      # Levenberg-Marquardt: a quadratic penalty around c on the prior node
    r = infer(model = cook_graph(mp = mp, Λp = Λp), data = (w = w,), meta = GRAPH_META,
              initialization = init_z(mp, Λp), iterations = bp_iters,
              returnvars = (z = KeepLast(),))
    m, V = mean_cov(r.posteriors[:z])
    return collect(m), Matrix(V)
end

"""
    solve_map_channels(channels, prior, d; c0, area_latent, iters, tol, verbose)

The Gauss-Newton solve over a channel registry on the graph engine
(`solve_channels_graph`: the graph owns the loop, the affine model
re-centred and the damping scaled from inside `infer`), alternated with
the AR node's noise identification when `noise` is given; stops on a
small step or on an objective decrease below `ftol` (the adaptive
solve's jitter floor is ~1e-3 in whitened units, so chasing it buys
nothing).
"""
function solve_map_channels(channels, prior, d::GateData; c0, area_latent::Int=0,
                            physics::Union{Nothing,Physics}=nothing,
                            noise::Union{Nothing,AbstractDict{Int,NoisePrior}}=nothing,
                            noise_rounds::Int=3, noise_tol::Float64=0.02,
                            levelled::Bool=false, level_w::Float64=0.05,
                            iters::Int=30, tol::Float64=1e-4, ftol::Float64=1e-3,
                            bp_iters::Int=20, verbose::Bool=false, trust::Bool=true)
    ph = physics === nothing ? Physics1D(area_latent) : physics
    solve_fixed = (chs, c) -> solve_channels_graph(chs, prior, d, ph; c0=c, iters, tol, ftol, bp_iters, verbose, trust)
    chs = collect(Channel, channels)
    noise_post = Dict{Int,NamedTuple}()
    c = copy(c0)
    res = nothing
    for round in 1:(noise === nothing ? 1 : noise_rounds)
        res = solve_fixed(chs, c)
        c = res.z
        noise === nothing && break
        if !res.converged
            verbose && println("    noise round $round skipped: the physics fit did not converge")
            break
        end
        S, J, lay = physics_outputs(c, d, ph)
        moved = 0.0
        for (i, np) in noise
            est = estimate_noise(chs[i], S, J, c, prior, lay, np)
            old_σ, old_ρ = chs[i].σ, chs[i].ρ
            moved = max(moved, abs(est.σ / old_σ - 1), abs(est.ρ - old_ρ))
            chs[i] = with_noise(unlevelled(chs[i]), est.σ, est.ρ)
            if levelled && count(noise_mask(chs[i])) >= 3     # the noise level as a process along each run, at the channel's own spacing
                runs = residual_runs(chs[i], S, J, c, prior, lay)
                levels = [volatility_level(e, clamp(est.post.ρ, 0.0, 0.999), est.σ; w=level_w).level for e in runs]   # ρ lies inside the stationary bound (its truncated summary); a negative correlation has no level process
                chs[i] = Levelled(chs[i], level_on_grid(noise_mask(chs[i]), levels))
            end
            noise_post[i] = (σ=est.σ, ρ=est.ρ, σ_sd=est.post.σ_sd, ρ_sd=est.post.ρ_sd)
            verbose && @printf("    noise round %d, channel %d (%s): σ %.3f -> %.3f (± %.3f), ρ %.3f -> %.3f (± %.3f)\n",
                               round, i, channel_label(chs[i]), old_σ, est.σ, est.post.σ_sd, old_ρ, est.ρ, est.post.ρ_sd)
        end
        verbose && flush(stdout)
        moved < noise_tol && break
    end
    return (; res..., channels=chs, noise=noise_post)
end

"""
    stream_fit(channels, prior, d, batch_edges; area_latent, c0, verbose) -> Vector

Refit at each time in `batch_edges` [s] on the readings up to that time
(static readings always), warm-started at the previous posterior mean.
Returns one record per batch: time, posterior mean, sd, covariance,
objective, outer iterations, wall.
"""
function stream_fit(channels, prior, d::GateData, batch_edges; area_latent::Int=0,
                    physics::Union{Nothing,Physics}=nothing, c0=nothing, verbose::Bool=true, kw...)
    c = c0 === nothing ? vcat(d.z0, z_center.(prior[(N_PHYS + 1):end])) : copy(c0)
    out = NamedTuple[]
    channels = collect(Channel, channels)        # the noise estimate of one batch carries to the next
    for (b, te) in enumerate(batch_edges)
        keep = d.t .<= te + 1e-9
        chs = [restrict(ch, keep) for ch in channels]
        t0 = time()
        res = solve_map_channels(chs, prior, d; c0=c, area_latent, physics, verbose=verbose, kw...)
        c = res.z
        for (i, np) in res.noise
            channels[i] = with_noise(unlevelled(channels[i]), np.σ, np.ρ)
            res.channels[i] isa Levelled && (channels[i] = Levelled(channels[i], res.channels[i].level))
        end
        sd = sqrt.(diag(res.cov))
        push!(out, (batch=b, t=te, z=res.z, sd=sd, cov=res.cov, F=res.F, channels=res.channels,
                    iters=length(res.hist), wall=time() - t0, misfit=res.misfit, noise=res.noise))
        verbose && @printf("  batch %2d  t = %5.1f h  F = %9.3f  outer %2d  (%.0f s)\n",
                           b, te / 3600, res.F, length(res.hist), time() - t0)
        verbose && flush(stdout)
    end
    return out
end
