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

export RealCookNoise, realcook_masks, realcook_readings, lid_windows, Instruments, read_at, clean_at, pit_sd_at, realcook_channels, realcook_noise_priors, lid_open_at

"The instruments' noise: per channel the marginal sd, the correlation TIME [s], and the sampling step [s]."
Base.@kwdef struct RealCookNoise
    pit_sd::Float64 = 1.5
    pit_tau::Float64 = 3000.0          # 50 min
    pit_step::Float64 = 120.0
    pit_add_bump::Float64 = 2.0        # the sd is (1 + bump·exp(−(t − t_add)/add_tau)) times after a fuel add
    add_tau::Float64 = 1800.0
    core_sd::Float64 = 1.2          # the calibration's floor for a meat probe; a mock below it makes the estimator discard the channel
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

"The grid indices inside a lid-open window after each event time (the wrap, every spritz up to the wrap or the pull, the pull, the fuel add)."
function lid_windows(t, events::NamedTuple, nc::RealCookNoise; t_end=Inf)
    open = falses(length(t))
    times = Float64[]
    ev = events
    haskey(ev, :t_wrap) && ev.t_wrap < 1e8 && push!(times, ev.t_wrap)
    haskey(ev, :t_pull) && ev.t_pull < 1e8 && push!(times, ev.t_pull)
    haskey(ev, :t_add) && ev.t_add < 1e8 && push!(times, ev.t_add)
    if haskey(ev, :t_spritz) && ev.t_spritz < 1e8
        tp = min(haskey(ev, :t_pull) && ev.t_pull < 1e8 ? ev.t_pull : t_end,
                 haskey(ev, :t_wrap) && ev.t_wrap < 1e8 ? ev.t_wrap : t_end)     # the train stops at the wrap, as the plant's does (no lid lift for a wrapped brisket)
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

# a unit AR(1) on the grid (per-step correlation from the correlation time); scaled by a time-varying level its
# marginal sd is exactly sd_t[k] at every sample
function _ar1_unit(rng, t, τ)
    dt = t[2] - t[1]
    ρ = exp(-dt / τ)
    a = zeros(length(t))
    a[1] = randn(rng)
    for k in 2:length(t)
        a[k] = ρ * a[k - 1] + sqrt(1 - ρ^2) * randn(rng)
    end
    return a
end

"""
    Instruments(t, nc, rng)

The mock's instruments on the grid: one unit AR(1) draw per channel at its correlation time (the pit, the core,
the humidity, the scale, in that order from `rng`), drawn once, so a cook read step by step (`read_at`) reads
exactly as one read whole (`realcook_readings`); and the instruments' model.
"""
struct Instruments
    nc::RealCookNoise
    t::Vector{Float64}
    unit::NamedTuple{(:pit, :core, :humidity, :mass),NTuple{4,Vector{Float64}}}
end
Instruments(t, nc::RealCookNoise, rng) = Instruments(nc, collect(Float64, t),
    (pit=_ar1_unit(rng, t, nc.pit_tau), core=_ar1_unit(rng, t, nc.core_tau), humidity=_ar1_unit(rng, t, nc.hum_tau), mass=_ar1_unit(rng, t, nc.mass_tau)))

"The pit probe's sd at time `t`: the level follows the fire, a bump after the fuel add."
pit_sd_at(nc::RealCookNoise, t, t_add) = nc.pit_sd * (t_add < 1e8 && t > t_add ? 1 + nc.pit_add_bump * exp(-(t - t_add) / nc.add_tau) : 1.0)

"Whether the lid is open at time `t` under the executed events (the wrap, the pull, the add, the spritz train to the wrap or the pull)."
function lid_open_at(t, events::NamedTuple, nc::RealCookNoise)
    ev = events
    inwin(te) = te < 1e8 && te <= t < te + nc.lid_open
    (haskey(ev, :t_wrap) && inwin(ev.t_wrap)) && return true
    (haskey(ev, :t_pull) && inwin(ev.t_pull)) && return true
    (haskey(ev, :t_add) && inwin(ev.t_add)) && return true
    if haskey(ev, :t_spritz) && ev.t_spritz < 1e8
        tp = min(haskey(ev, :t_pull) && ev.t_pull < 1e8 ? ev.t_pull : Inf, haskey(ev, :t_wrap) && ev.t_wrap < 1e8 ? ev.t_wrap : Inf)
        n = floor(Int, (t - ev.t_spritz) / ev.dt_spritz)
        n >= 0 && ev.t_spritz + n * ev.dt_spritz <= tp && inwin(ev.t_spritz + n * ev.dt_spritz) && return true
    end
    return false
end

"The clean instrument values of a monolith output at save index `k`: the pit, the core probe at `depth` from the top, the vapor pressure, the mass on the scale."
function clean_at(o, k::Int; A, L, depth, cfg::MeatConfig=MeatConfig())
    nx = size(o.T_field, 1)
    w = fv_weights(nx)
    pos = 1 + (1 - depth / L) * (nx - 1)
    k0 = clamp(floor(Int, pos), 1, nx - 1); f = pos - k0
    return (pit=o.Tpit[k], core=(1 - f) * o.T_field[k0, k] + f * o.T_field[k0 + 1, k], humidity=o.pv_pit[k],
            mass=meat_mass_1d(dot(w, view(o.W_field, :, k)), dot(w, view(o.F_field, :, k)), A, L, cfg))
end

"""
    read_at(inst, k, clean, events) -> (readings, present)

The instruments' readings at grid point `k` from the plant's clean values there (`clean_at`) under the executed
`events`: each channel at its own rate, none while the lid is open, the pit's noise at the fire's level, the
humidity capped. `present` says which channels sampled.
"""
function read_at(inst::Instruments, k::Int, clean::NamedTuple, events::NamedTuple)
    nc = inst.nc; t = inst.t; dt = t[2] - t[1]
    at_rate(step) = (k - 1) % max(1, Int(round(step / dt))) == 0
    open = lid_open_at(t[k], events, nc)
    t_add = haskey(events, :t_add) ? events.t_add : Inf
    present = (pit=at_rate(nc.pit_step) && !open, core=at_rate(nc.core_step) && !open, humidity=at_rate(nc.hum_step) && !open, mass=at_rate(nc.mass_step) && !open)
    readings = (pit=clean.pit + pit_sd_at(nc, t[k], t_add) * inst.unit.pit[k], core=clean.core + nc.core_sd * inst.unit.core[k],
                humidity=min(clean.humidity + nc.hum_sd * inst.unit.humidity[k], nc.hum_cap), mass=clean.mass + nc.mass_sd * inst.unit.mass[k])
    return readings, present
end

"""
    realcook_readings(o, t, events, nc, rng; A, L, depth) -> (readings, clean, masks, open)

The instruments' readings from a monolith output `o` on the grid `t`
under the executed `events`: the pit probe (its sd bumped after the
fuel add), the core probe at `depth` from the top, the humidity sensor
(capped), and the composition-conserving scale mass; each channel's mask
and the lid-open windows. The same numbers `read_at` gives step by step.
"""
function realcook_readings(o, t, events::NamedTuple, nc::RealCookNoise, rng; A, L, depth, cfg::MeatConfig=MeatConfig())
    open = lid_windows(t, events, nc)
    masks = realcook_masks(t, open, nc)
    inst = Instruments(t, nc, rng)
    n = length(t)
    pit = zeros(n); core = zeros(n); hum = zeros(n); mass = zeros(n)
    cpit = zeros(n); ccore = zeros(n); chum = zeros(n); cmass = zeros(n)
    for k in 1:n
        c = clean_at(o, k; A, L, depth, cfg)
        r, _ = read_at(inst, k, c, events)
        pit[k], core[k], hum[k], mass[k] = r.pit, r.core, r.humidity, r.mass
        cpit[k], ccore[k], chum[k], cmass[k] = c.pit, c.core, c.humidity, c.mass
    end
    return (pit=pit, core=core, humidity=hum, mass=mass), (pit=cpit, core=ccore, humidity=chum, mass=cmass), masks, open
end

"""
    realcook_noise_priors(cal_path, dt) -> Dict{Int,NoisePrior}

The four real-cook channels' noise priors on a grid of step `dt` [s]: the calibration's posterior for the pit
and the core probes (`noise_prior_real.json`, identified on a ten-minute grid, its ρ rescaled to the step), the
instruments' spec for the humidity sensor and the scale, with floors (a sparse channel's sd reads low from a few
samples). A file written by `next_cook.py` carries the humidity and scale channels too, pooled across the
cooks flown, and its own floors; they are used when present.
"""
function realcook_noise_priors(cal_path::AbstractString, dt::Real)
    j = JSON.parsefile(cal_path)
    cal = noise_priors_from_file(cal_path)
    grid = Float64(get(j, "grid_s", 600.0))
    tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
    floors = get(j, "floors", Dict())
    withfloor(np, name, default) = NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean, np.ρ_sd; floor=Float64(get(floors, name, default)))
    hum = haskey(cal, "humidity") ? tostep(withfloor(cal["humidity"], "humidity", 60.0), grid) : NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0)
    mass = haskey(cal, "mass") ? tostep(withfloor(cal["mass"], "mass", 0.015), grid) : NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015)
    return Dict(1 => tostep(cal["pit"], grid), 2 => tostep(cal["core"], grid), 3 => hum, 4 => mass)
end

"""
    realcook_channels(readings, masks, np; cap, depth_latent) -> Vector{Channel}

The registry's channels for the real-cook instruments, in the order the noise priors index them: the pit probe,
the meat probe at the latent bounded depth fraction, the censored humidity sensor, the scale; the readings and
masks on the full grid (a mock filled ahead, or arrays a pilot fills as the samples arrive).
"""
function realcook_channels(readings, masks, np::AbstractDict{Int,NoisePrior}; cap::Real, depth_latent::Int)
    return Channel[PitProbe(np[1].σ_mean, np[1].ρ_mean, Bool.(masks.pit); data=Float64.(readings.pit)),
                   MeatProbe(np[2].σ_mean, np[2].ρ_mean, Bool.(masks.core); depth_latent, data=Float64.(readings.core)),
                   CensoredHumidity(np[3].σ_mean, np[3].ρ_mean, Bool.(masks.humidity); cap=Float64(cap), data=Float64.(readings.humidity)),
                   MassScale(np[4].σ_mean, np[4].ρ_mean, Bool.(masks.mass); data=Float64.(readings.mass))]
end
