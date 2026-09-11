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
restrict(ch::MassScale, keep) = MassScale(ch.σ, ch.ρ, ch.mask .& keep, ch.shape_latent, ch.data)
restrict(ch::MeatProbe, keep) = MeatProbe(ch.s_fixed, ch.depth_fixed, ch.depth_latent, ch.σ, ch.ρ,
                                          ch.mask .& keep, ch.data)
restrict(ch::FuelScale, keep) = FuelScale(ch.σ, ch.ρ, ch.mask .& keep, ch.data)
restrict(ch::BarkLightness, keep) = BarkLightness(ch.σ, ch.ρ, ch.mask .& keep, ch.data)
restrict(ch::WetBulb, keep) = WetBulb(ch.σ, ch.ρ, ch.mask .& keep, ch.p_atm, ch.a_psy, ch.data)
restrict(ch::StaticReading, keep) = ch

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

_objective(a, w, z, prior) = 0.5 * sum(abs2, w .- a) + prior_terms(z, prior)[1]

# the objective at a trial point from solves alone; a solve that fails (the
# adaptive monolith raises on an unstable step, the 2-D engine returns NaN)
# is an infinite objective, which the line search treats as a worse step
function _trial_objective(channels, c_new, d, ph, J0, prior, lay, w)
    Sn = try
        physics_stack(c_new, d, ph)[1]
    catch err
        err isa InterruptException && rethrow()
        return Inf
    end
    an, _, _ = assemble(channels, c_new, Sn, J0, prior, lay)
    return _objective(an, w, c_new, prior)
end

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
                            physics::Union{Nothing,Physics}=nothing,
                            noise::Union{Nothing,AbstractDict{Int,NoisePrior}}=nothing,
                            noise_rounds::Int=3, noise_tol::Float64=0.02,
                            levelled::Bool=false, level_w::Float64=0.05,
                            engine::Symbol=:graph,
                            iters::Int=30, tol::Float64=1e-4, ftol::Float64=1e-3,
                            bp_iters::Int=20, verbose::Bool=false)
    ph = physics === nothing ? Physics1D(area_latent) : physics
    engine in (:graph, :outer) || throw(ArgumentError("engine must be :graph (the loop inside infer) or :outer (the loop around it)"))
    solve_fixed = engine === :graph ?
        (chs, c) -> solve_channels_graph(chs, prior, d, ph; c0=c, iters, tol, ftol, bp_iters, verbose) :
        (chs, c) -> _solve_fixed_noise(chs, prior, d, ph; c0=c, iters, tol, ftol, bp_iters, verbose)
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
                levels = [volatility_level(e, clamp(est.post.ρ, 0.0, 0.999), est.σ; w=level_w).level for e in runs]   # the AR posterior mean can exceed 1 on a drifting residual
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

function _solve_fixed_noise(channels, prior, d::GateData, ph::Physics; c0, iters, tol, ftol, bp_iters, verbose)
    c = copy(c0)
    S, J, lay = physics_outputs(c, d, ph)
    a, B, w = assemble(channels, c, S, J, prior, lay)
    any(isnan, w) && throw(ArgumentError("every channel in a fit must carry data"))
    Fc = _objective(a, w, c, prior)
    hist = NamedTuple[]
    m, V = c, zeros(length(c), length(c))
    J0 = zeros(size(J))          # a zero Jacobian: predictions only, in the line search
    λ = 0.0                      # Levenberg-Marquardt damping: zero (pure Gauss-Newton) until a step is rejected
    λ0 = 0.1
    converged = false
    for k in 1:iters
        t0 = time()
        # a step rejected at the line search's floor, or absurd in size, raises the damping and retries
        # on the same Jacobian; a full step accepted lowers it
        local step, α, Fn, c_new
        accepted = false
        for attempt in 1:8
            m, V = graph_posterior_affine(c, a, B, w, prior; bp_iters, damping=λ)
            step = m - c
            α = 1.0
            c_new = c + step
            Fn = norm(step) > 20 ? Inf : _trial_objective(channels, c_new, d, ph, J0, prior, lay, w)
            worse(F) = !isfinite(F) || F > Fc          # a solve that blew up is a worse step, not a NaN to compare
            while worse(Fn) && α > 1e-3
                α /= 2
                c_new = c + α * step
                Fn = _trial_objective(channels, c_new, d, ph, J0, prior, lay, w)
            end
            if !worse(Fn)
                accepted = true
                α == 1.0 && (λ = λ / 3 < λ0 / 10 ? 0.0 : λ / 3)
                break
            end
            if norm(step) < 10 * tol     # no decrease from a negligible step: at the jitter floor, converged
                converged = true
                break
            end
            λ = λ == 0.0 ? λ0 : 10λ
            verbose && @printf("    it %2d  rejected (|step|=%.3g); damping -> %.3g\n", k, norm(step), λ)
        end
        push!(hist, (it=k, F=Fc, step=norm(step), alpha=α, wall=time() - t0, damping=λ))
        verbose && @printf("    it %2d  F=%10.4f -> %10.4f  |step|=%.3g  alpha=%.3g  damping=%.2g  (%.1f s)\n",
                           k, Fc, Fn, norm(step), α, λ, time() - t0)
        verbose && flush(stdout)
        accepted || break                      # converged at the floor, or stuck: either way the point stands
        converged = (α * norm(step) < tol) || (Fc - Fn < ftol)   # step or objective floor
        c, Fc = c_new, Fn
        S, J, _ = physics_outputs(c, d, ph)     # the Jacobian once per accepted step
        a, B, _ = assemble(channels, c, S, J, prior, lay)
        converged && break
    end
    m, V = graph_posterior_affine(c, a, B, w, prior; bp_iters)     # the reported posterior is undamped
    mis = misfit(channels, a, B, w, V)
    return (z=c, mean=m, cov=V, F=Fc, hist=hist, misfit=mis, converged=converged)
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
    c = c0 === nothing ? vcat(d.z0, [isbeta(p) ? log(p.p1 / p.p2) : p.p1 for p in prior[(N_PHYS + 1):end]]) :
        copy(c0)
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
