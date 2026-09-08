# A plant that reads like a real cook. Included by BrisketEstimator
# after horizon.jl; the physics is any monolith output on a fine grid,
# the readings are what instruments would give:
#   - each channel at its own rate (a mask on the grid)
#   - the noise an AR(1) per channel at its own correlation time, with a
#     level that follows the cook: the pit probe's sd triples after a
#     fuel addition and decays back over half an hour (the fire regime)
#   - lid lifts (the wrap, each spritz, the pull) drop every channel's
#     samples for the minutes the lid is open
#   - the humidity sensor saturates: its reading is capped at the vapor
#     pressure its housing can hold
#   - the scale reads every half hour and drops the lid-open samples
# and the estimator's channels for those readings, with the censored
# humidity as its own channel.

export RealCookNoise, realcook_masks, realcook_readings, CensoredHumidity, lid_windows

"The instruments' noise: per channel the marginal sd, the correlation TIME [s], and the sampling step [s]."
Base.@kwdef struct RealCookNoise
    pit_sd::Float64 = 1.5
    pit_tau::Float64 = 3000.0          # 50 min
    pit_step::Float64 = 120.0
    pit_add_bump::Float64 = 2.0        # the sd is (1 + bump·exp(−(t − t_add)/add_tau)) times after a fuel add
    add_tau::Float64 = 1800.0
    core_sd::Float64 = 0.5
    core_tau::Float64 = 12000.0        # 3.3 h
    core_step::Float64 = 120.0
    hum_sd::Float64 = 100.0
    hum_tau::Float64 = 2700.0          # 45 min
    hum_step::Float64 = 600.0
    hum_cap::Float64 = 6500.0          # the sensor housing's saturation [Pa]
    mass_sd::Float64 = 0.02
    mass_tau::Float64 = 2600.0
    mass_step::Float64 = 1800.0
    lid_open::Float64 = 360.0          # the lid is open six minutes at each event
end

"The grid indices inside a lid-open window after each event time (the wrap, every spritz, the pull, the fuel add)."
function lid_windows(t, events::NamedTuple, nc::RealCookNoise; t_end=Inf)
    open = falses(length(t))
    times = Float64[]
    ev = events
    haskey(ev, :t_wrap) && ev.t_wrap < 1e8 && push!(times, ev.t_wrap)
    haskey(ev, :t_pull) && ev.t_pull < 1e8 && push!(times, ev.t_pull)
    haskey(ev, :t_add) && ev.t_add < 1e8 && push!(times, ev.t_add)
    if haskey(ev, :t_spritz) && ev.t_spritz < 1e8
        tp = haskey(ev, :t_pull) && ev.t_pull < 1e8 ? ev.t_pull : t_end
        ts = ev.t_spritz
        while ts <= min(tp, t[end])
            push!(times, ts)
            ts += ev.dt_spritz
        end
    end
    for te in times
        open .|= (t .>= te) .& (t .< te + nc.lid_open)
    end
    return open
end

"Each channel's mask on the grid: its own rate, minus the lid-open windows."
function realcook_masks(t, open, nc::RealCookNoise)
    dt = t[2] - t[1]
    every(step) = [(k - 1) % max(1, Int(round(step / dt))) == 0 for k in eachindex(t)]
    return (pit=every(nc.pit_step) .& .!open, core=every(nc.core_step) .& .!open,
            humidity=every(nc.hum_step) .& .!open, mass=every(nc.mass_step) .& .!open)
end

# a unit AR(1) on the grid (per-step correlation from the correlation time) scaled by a
# time-varying level: the marginal sd is exactly sd_t[k] at every sample
function _ar1_level(rng, t, sd_t, τ)
    dt = t[2] - t[1]
    ρ = exp(-dt / τ)
    a = zeros(length(t))
    a[1] = randn(rng)
    for k in 2:length(t)
        a[k] = ρ * a[k - 1] + sqrt(1 - ρ^2) * randn(rng)
    end
    return sd_t .* a
end

"""
    realcook_readings(o, t, events, nc, rng; A, L, depth) -> (readings, clean, masks, open)

The instruments' readings from a monolith output `o` on the grid `t`
under the executed `events`: the pit probe (its sd bumped after the
fuel add), the core probe at `depth` from the top, the humidity sensor
(capped), the scale (ρ_d A L (1 + W̄ + F̄)); each channel's mask and the
lid-open windows.
"""
function realcook_readings(o, t, events::NamedTuple, nc::RealCookNoise, rng; A, L, depth)
    open = lid_windows(t, events, nc)
    masks = realcook_masks(t, open, nc)
    nx = size(o.T_field, 1)
    w = fv_weights(nx)
    pos = 1 + (1 - depth / L) * (nx - 1)
    k0 = clamp(floor(Int, pos), 1, nx - 1); f = pos - k0
    core = [(1 - f) * o.T_field[k0, k] + f * o.T_field[k0 + 1, k] for k in axes(o.T_field, 2)]
    mass = [RHO_DRY_EST * A * L * (1 + dot(w, view(o.W_field, :, k)) + dot(w, view(o.F_field, :, k))) for k in axes(o.W_field, 2)]
    clean = (pit=collect(o.Tpit), core=core, humidity=collect(o.pv_pit), mass=mass)
    # the pit's level follows the fire: a bump after the fuel add
    pit_sd = fill(nc.pit_sd, length(t))
    if haskey(events, :t_add) && events.t_add < 1e8
        for k in eachindex(t)
            t[k] > events.t_add && (pit_sd[k] *= 1 + nc.pit_add_bump * exp(-(t[k] - events.t_add) / nc.add_tau))
        end
    end
    pit = clean.pit .+ _ar1_level(rng, t, pit_sd, nc.pit_tau)
    core_r = clean.core .+ _ar1_level(rng, t, fill(nc.core_sd, length(t)), nc.core_tau)
    hum = min.(clean.humidity .+ _ar1_level(rng, t, fill(nc.hum_sd, length(t)), nc.hum_tau), nc.hum_cap)
    mass_r = clean.mass .+ _ar1_level(rng, t, fill(nc.mass_sd, length(t)), nc.mass_tau)
    return (pit=pit, core=core_r, humidity=hum, mass=mass_r), clean, masks, open
end

"""
    CensoredHumidity(σ, ρ, mask; cap, data)

The humidity sensor with a saturation cap: a reading at the cap says
only that the vapor pressure was at least that, so its prediction is
min(pred, cap), smooth (a softplus of width `s`), and its residual is
zero whenever the model is above the cap.
"""
struct CensoredHumidity <: Channel
    σ::Float64
    ρ::Float64
    mask::Vector{Bool}
    cap::Float64
    s::Float64
    data::Union{Nothing,Vector{Float64}}
end
CensoredHumidity(σ, ρ, mask; cap=6500.0, s=50.0, data=nothing) = CensoredHumidity(σ, ρ, collect(Bool, mask), cap, s, data)
function predict(ch::CensoredHumidity, S, J, z, prior, lay)
    idx = (lay.opv + 1):(lay.opv + lay.nt)
    r = copy(S[idx]); Jr = copy(J[idx, :])
    y = ch.data
    for k in 1:lay.nt
        if y !== nothing && y[k] >= ch.cap - 1e-9
            # censored: min(r, cap) as cap − softplus(cap − r); its slope is logistic((cap − r)/s): one below the cap, zero above
            x = (ch.cap - r[k]) / ch.s
            r[k] = ch.cap - ch.s * (x > 30 ? x : log1p(exp(x)))
            Jr[k, :] .*= 1 / (1 + exp(-x))
        end
    end
    return r, Jr
end
noise_mask(ch::CensoredHumidity) = ch.data === nothing ? ch.mask : ch.mask .& (ch.data .< ch.cap - 1e-9)
function extra_rows(ch::CensoredHumidity, level)
    censored = findall(ch.mask .& .!noise_mask(ch))
    rows = zeros(length(censored), length(ch.mask))
    for (i, k) in enumerate(censored)
        rows[i, k] = 1 / (ch.σ * level[k])
    end
    return rows
end
# the AR rows over the uncensored samples (the chain hops over a censored stretch as over a gap), then one
# independent row per censored sample: "the true value was at least the cap", weighted like a reading
whitening(ch::CensoredHumidity) = vcat(whitening_rows(noise_mask(ch), ch.σ, ch.ρ), extra_rows(ch, ones(length(ch.mask))))
misfit_rows(ch::CensoredHumidity) = size(whitening_rows(noise_mask(ch), ch.σ, ch.ρ), 1)
channel_label(::CensoredHumidity) = "humidity (censored)"
restrict(ch::CensoredHumidity, keep) = CensoredHumidity(ch.σ, ch.ρ, ch.mask .& keep, ch.cap, ch.s, ch.data)
with_noise(ch::CensoredHumidity, σ, ρ) = CensoredHumidity(σ, ρ, ch.mask, ch.cap, ch.s, ch.data)
