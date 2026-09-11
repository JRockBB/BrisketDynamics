# --- the estimator as a filter: predict, then update on the window ------------------------------
#
# The streaming estimator refits every latent as a constant on the full history. That is exact when the
# plant's parameters are constants, and wrong when they move: a new charge of fuel at the add has its own
# burn rate and enthalpy, the leak drifts as the vents are nudged, the surface coefficients change as the
# bark sets; a constant refit averages a parameter that has moved and reports the width of the average.
# Here each batch is a prediction and an update. The prediction: the last posterior, its covariance inflated
# by process noise (a variance rate per hour per latent; zero keeps a latent a constant), as a full Gaussian
# prior in z (FilterPrior). The update: the window's readings alone, the physics solved over the window from
# the state the last fit's solve handed off (segments.jl), with a0 the handed-off ignited fraction (a state,
# not a parameter: its prior is that value, tight). The first batch is the batch fit from the prior families.
# What is approximated: the handed-off state is taken as known (its own uncertainty from the last window's
# parameter error is not carried), which is what the process noise has to cover and what the coverage test
# on a drifting plant judges.

const A0_INDEX = N_THETA + 7          # φ[7], the ignited fraction: the only place a0 acts is the initial state

"""
    filter_fit(readings, masks, t, u, pit, cfg, sites, names, events, edges;
               q, noise, cap, depth_latent, area_latent, a0_sd=0.05, verbose) -> Vector{NamedTuple}

The real-cook channels (`readings`/`masks`: pit, core, humidity, mass on the grid `t`) streamed as a filter
with batch edges `edges` [s]. `q` is the process-noise variance rate per hour per latent (length of `sites`,
in z²/h). `noise` are the channels' NoisePrior by index (identified per window as in `solve_map_channels`,
the estimate carried to the next window). Returns one record per window: `t`, `z`, `cov`, `sd`, `F`,
`misfit`, `noise`, `iters`, `wall`, `prior_m`, `prior_sd` (the prediction the window started from).
"""
function filter_fit(readings, masks, t, u, pit::PitConfig, cfg::MeatConfig, sites::Vector{PriorSite}, names::Vector{String},
                    events::NamedTuple, edges; q::AbstractVector, noise::AbstractDict{Int,NoisePrior}, cap::Float64,
                    depth_latent::Int, area_latent::Int, a0_sd::Float64=0.05, verbose::Bool=false, iters::Int=30)
    n = length(sites)
    sites_phys = sites[1:N_PHYS]
    z = vcat([isbeta(p) ? log(p.p1 / p.p2) : p.p1 for p in sites_phys], [isbeta(p) ? log(p.p1 / p.p2) : p.p1 for p in sites[(N_PHYS + 1):end]])
    V = nothing
    carried = Dict(i => (np.σ_mean, np.ρ_mean) for (i, np) in noise)     # the noise estimate carried across windows
    levels = Dict{Int,Union{Nothing,Vector{Float64}}}(i => nothing for i in keys(noise))
    ics = (;)
    pit_w = pit                 # the pit's state travels in `ics`; the config is the cook's
    a0_hand = NaN
    i_start = 1
    out = NamedTuple[]
    for (b, te) in enumerate(edges)
        t0 = time()
        i_end = findlast(t .<= te + 1e-9)
        idx = i_start:i_end
        tw, uw = t[idx], u[idx]
        Δh = (t[i_end] - t[i_start]) / 3600
        # the window's channels: the readings on the window, the carried noise, the level process where one was inferred
        base = [PitProbe(carried[1]..., Bool.(masks.pit[idx]); data=Float64.(readings.pit[idx])),
                MeatProbe(carried[2]..., Bool.(masks.core[idx]); depth_latent, data=Float64.(readings.core[idx])),
                CensoredHumidity(carried[3]..., Bool.(masks.humidity[idx]); cap, data=Float64.(readings.humidity[idx])),
                MassScale(carried[4]..., Bool.(masks.mass[idx]); data=Float64.(readings.mass[idx]))]
        chs = [levels[i] === nothing ? ch : Levelled(ch, levels[i][idx]) for (i, ch) in enumerate(base)]
        d = estimator_setup(tw, uw, pit_w, cfg, sites_phys, names)
        ph = Physics1D(area_latent=area_latent, events=merge(events, ics))
        # the prediction: the prior families for the first window; then the last posterior inflated by the
        # process noise, a0 reset to the handed-off ignited fraction
        if V === nothing
            prior_w = sites
            pm, psd = copy(z), fill(NaN, n)
        else
            Vp = Matrix(Symmetric(V)) + Diagonal(q .* Δh)
            m = copy(z)
            m[A0_INDEX] = log(a0_hand / (1 - a0_hand))
            Vp[A0_INDEX, :] .= 0.0; Vp[:, A0_INDEX] .= 0.0; Vp[A0_INDEX, A0_INDEX] = a0_sd^2
            prior_w = FilterPrior(sites, m, inv(Symmetric(Vp)))
            pm, psd = m, sqrt.(diag(Vp))
        end
        res = solve_map_channels(chs, prior_w, d; c0=pm, physics=ph, engine=:graph, noise=noise, noise_rounds=1,
                                 levelled=true, iters, verbose)
        for (i, v) in res.noise
            carried[i] = (v.σ, v.ρ)
            lv = res.channels[i] isa Levelled ? res.channels[i].level : nothing
            if lv !== nothing          # the level lives on the window's grid; keep it on the full grid for the next window
                full = levels[i] === nothing ? ones(length(t)) : levels[i]
                full[idx] = lv
                levels[i] = full
            end
        end
        z = res.z
        V = Matrix(Symmetric(res.cov))
        # the hand-off: the window solved at the MAP, its end state seeds the next window
        x = from_z(z[1:N_PHYS], sites_phys)
        φ = x[(N_THETA + 1):N_PHYS]
        o = monolith_solve(x[1:N_THETA], φ, uw, tw, tw; pit=pit_w, cfg, A_meat=exp(z[area_latent]), abstol=d.abstol, reltol=d.reltol, ics..., events...)
        ics, a0_hand = handoff(o, length(tw))
        i_start = i_end
        push!(out, (t=t[i_end], z=copy(z), cov=copy(V), sd=sqrt.(max.(diag(V), 0.0)), F=res.F, misfit=res.misfit, noise=res.noise,
                    iters=length(res.hist), wall=time() - t0, prior_m=pm, prior_sd=psd, converged=res.converged, window_h=Δh))
        verbose && @printf("  filter window %d to %.1f h: %d iters, F %.2f, %.0f s\n", b, te / 3600, length(res.hist), res.F, time() - t0)
    end
    return out
end
