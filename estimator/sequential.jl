# --- the filter on the graph: the belief over the plant's state, the latents and the instruments' discrepancies,
# --- updated as each measurement arrives ----------------------------------------------------------------------
#
# The latent block is [z (the latents); x (the plant's state at the last update: the meat's temperature, water,
# collagen, fat and salt fields, the bark and the ring, the pit's temperature, fuel, pan, firebox and ignited
# fraction, as `handoff` carries them); d (one AR(1) discrepancy per channel: the coloured part of its error, the
# calibration's noise model as a state)]. An update at grid point k is the same graph fit as every other
# estimator here: `solve_map_channels` on the graph engine over the registry's channels restricted to the
# samples at k and wrapped as `Sequential` (the reading is the physics plus the channel's discrepancy plus white
# noise at the instrument's floor), one node per residual, the censored humidity's one-sided rows, the physics
# linearized and re-centred by the iterated loop, under a full-covariance Gaussian prior (`FilterPrior`) that is
# the belief carried to k: the discrepancies through their AR(1) transitions (the level in the innovation), the
# state through the physics with the transition's Jacobian from the same ForwardDiff pass the graph linearizes
# with (`PhysicsState`), the latents with their process noise. Where no sample arrives the belief is only
# propagated. Nothing here runs on a clock: the update happens when a sample arrives; a channel's noise (σ, ρ)
# is re-identified on the AR node when its discrepancy series has grown by one correlation time; its level (the
# GCV volatility smoother on its standardized innovations, against the other channels') whenever it delivers
# a sample; a level over the threshold, or a reading frozen at its last value, is a sensor the belief has
# stopped trusting (its discrepancy innovation inflated, a fault flag for the record).
#
# The belief after every update, the pit, centre and surface temperatures with their widths and the plant's
# state for the plan, is what the pilot plans on and what a figure of the cook must draw.

"The layout of the joint block: the latents first (the channels' latent indices unchanged), the state, then one discrepancy per channel."
struct FilterLayout
    nx::Int              # meat nodes
    n_state::Int         # 5 nx fields + bark_b, bark_s, ring + Tpit, m_fuel, m_pan, T_pan, T_fb, ignited
    n_z::Int
    n_ch::Int
    n::Int
end
FilterLayout(nx::Int, n_z::Int, n_ch::Int=0) = FilterLayout(nx, 5 * nx + 9, n_z, n_ch, n_z + 5 * nx + 9 + n_ch)
const STATE_SCALARS = (:B0, :S0, :R0, :Tpit0, :mfuel0, :mpan0, :Tpan0, :Tfb0)     # after the five fields; then the ignited fraction
state_range(lay::FilterLayout) = (lay.n_z + 1):(lay.n_z + lay.n_state)
disc_range(lay::FilterLayout) = (lay.n_z + lay.n_state + 1):lay.n
disc_index(lay::FilterLayout, c::Int) = lay.n_z + lay.n_state + c
"The block index of the pit's temperature, the centre node and the surface node of the temperature field (the probe position formula: depth 0 is node nx, the centre node 1)."
pit_index(lay::FilterLayout) = lay.n_z + 5 * lay.nx + 4
centre_index(lay::FilterLayout) = lay.n_z + 1
surface_index(lay::FilterLayout) = lay.n_z + lay.nx
ignited_index(lay::FilterLayout) = lay.n_z + lay.n_state
"The state block's sites: Gaussian in the raw coordinate (`from_z` is the identity on the \"linear\" family)."
state_sites(n::Int) = [PriorSite("linear", 0.0, 1.0) for _ in 1:n]
"The discrepancy sites: the same linear family, in the instrument's units."
disc_sites(n::Int) = state_sites(n)

"The state vector from `handoff`'s initial conditions and the ignited fraction."
pack_state(ics::NamedTuple, a, lay::FilterLayout) =
    vcat(ics.T0f, ics.W0f, ics.C0f, ics.F0f, ics.Ssalt0f, [getfield(ics, k) for k in STATE_SCALARS], [a])
"The initial-condition keywords and the ignited fraction from a state vector (any element type)."
function unpack_state(x, lay::FilterLayout)
    nx = lay.nx
    f(i) = x[((i - 1) * nx + 1):(i * nx)]
    sc = Dict(k => x[5 * nx + j] for (j, k) in enumerate(STATE_SCALARS))
    ics = (T0f=f(1), W0f=f(2), C0f=f(3), F0f=f(4), Ssalt0f=f(5), B0=sc[:B0], S0=sc[:S0], R0=sc[:R0],
           Tpit0=sc[:Tpit0], mfuel0=sc[:mfuel0], mpan0=sc[:mpan0], Tpan0=sc[:Tpan0], Tfb0=sc[:Tfb0])
    return ics, x[5 * nx + 9]
end

"""
    StateNoise(; T, W, C, F, S, bark, T_pit, m_fuel, m_pan, T_pan, T_fb, ignited, T0, W0, ...)

The state's process noise and initial widths by field group, in the field's own units: a rate per √hour for
the temperature field [K], the water [kg/kg], the collagen, fat and salt fields, the bark and ring scalars, the
pit's temperature [K], fuel [kg], pan mass [kg] and temperature [K], the firebox [K] and the ignited fraction;
and the sd of each at the start (the recipe's meat and pit known to about a kelvin, the loads to a few
percent). The rates are what the consistency bench earns (NEES of the temperatures and of the latents in their
bands); the defaults are a starting point, not a result.
"""
Base.@kwdef struct StateNoise
    T::Float64 = 0.5
    W::Float64 = 0.01
    C::Float64 = 0.005
    F::Float64 = 0.005
    S::Float64 = 0.001
    bark::Float64 = 0.01
    T_pit::Float64 = 2.0
    m_fuel::Float64 = 0.05
    m_pan::Float64 = 0.02
    T_pan::Float64 = 2.0
    T_fb::Float64 = 2.0
    ignited::Float64 = 0.02
    T0::Float64 = 1.0
    W0::Float64 = 0.05
    C0::Float64 = 1e-3
    F0::Float64 = 1e-3
    S0::Float64 = 1e-3
    bark0::Float64 = 1e-3
    T_pit0::Float64 = 2.0
    m_fuel0::Float64 = 0.1
    m_pan0::Float64 = 0.05
    T_pan0::Float64 = 2.0
    T_fb0::Float64 = 2.0
    ignited0::Float64 = 0.05
end
function _state_vec(lay::FilterLayout, fields, scalars, ign)
    nx = lay.nx
    return vcat([fill(Float64(v), nx) for v in fields]..., collect(Float64, scalars), [Float64(ign)])
end
"The process-noise rate of every state entry [units/√h]."
state_rates(sn::StateNoise, lay::FilterLayout) =
    _state_vec(lay, (sn.T, sn.W, sn.C, sn.F, sn.S), (sn.bark, sn.bark, sn.bark, sn.T_pit, sn.m_fuel, sn.m_pan, sn.T_pan, sn.T_fb), sn.ignited)
"The initial sd of every state entry."
state_init_sd(sn::StateNoise, lay::FilterLayout) =
    _state_vec(lay, (sn.T0, sn.W0, sn.C0, sn.F0, sn.S0), (sn.bark0, sn.bark0, sn.bark0, sn.T_pit0, sn.m_fuel0, sn.m_pan0, sn.T_pan0, sn.T_fb0), sn.ignited0)

"The groups of the state block, in `StateNoise`'s order: one rate and one initial width each."
const STATE_GROUPS = ("T", "W", "C", "F", "S", "bark", "T_pit", "m_fuel", "m_pan", "T_pan", "T_fb", "ignited")
"""
    state_groups(lay) -> Vector{Pair{String,UnitRange{Int}}}

Each group's name and the range of its entries within the state vector (the five fields of `nx` nodes, the
three bark and ring scalars, the five pit scalars, the ignited fraction); the bench scores the belief's
consistency group by group and earns the rates from it.
"""
function state_groups(lay::FilterLayout)
    nx = lay.nx
    r = Pair{String,UnitRange{Int}}[]
    for (i, g) in enumerate(("T", "W", "C", "F", "S")); push!(r, g => ((i - 1) * nx + 1):(i * nx)); end
    push!(r, "bark" => (5 * nx + 1):(5 * nx + 3))
    for (j, g) in enumerate(("T_pit", "m_fuel", "m_pan", "T_pan", "T_fb")); push!(r, g => (5 * nx + 3 + j):(5 * nx + 3 + j)); end
    push!(r, "ignited" => (5 * nx + 9):(5 * nx + 9))
    return r
end

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
"The state-noise file the bench writes and every driver reads."
const STATE_NOISE_FILE = joinpath(REPO_ROOT, "data", "calibration", "state_noise.json")
"The filter's own sources, fingerprinted with the physics: a change to any of them means the rates must be re-earned."
const FILTER_FILES = ("sequential.jl", "channels.jl", "noise.jl", "volatility.jl", "realcook.jl", "drift.jl", "segments.jl", "iterated.jl")

"""
    physics_fingerprint(; repo=REPO_ROOT) -> String

The first twelve hex digits of the SHA-1 over the physics sources (`src/**/*.jl`) and the filter's own files
(`FILTER_FILES`), each as its repository path and its bytes with carriage returns dropped, in path order. The
state-noise file records the fingerprint it was earned on; `load_state_noise` warns when the tree's differs.
"""
function physics_fingerprint(; repo::AbstractString=REPO_ROOT)
    files = String[]
    for (root, _, fs) in walkdir(joinpath(repo, "src")), f in fs
        endswith(f, ".jl") && push!(files, replace(relpath(joinpath(root, f), repo), "\\" => "/"))
    end
    append!(files, ["estimator/" * f for f in FILTER_FILES])
    sort!(files)
    ctx = SHA.SHA1_CTX()
    for f in files
        SHA.update!(ctx, codeunits(f))
        SHA.update!(ctx, filter(!=(0x0d), read(joinpath(repo, f))))
    end
    return bytes2hex(SHA.digest!(ctx))[1:12]
end

"The `StateNoise` as the file's fields: the rates and the initial sds by group."
state_noise_dict(sn::StateNoise) = Dict("rates" => Dict(g => getfield(sn, Symbol(g)) for g in STATE_GROUPS),
                                        "init_sd" => Dict(g => getfield(sn, Symbol(g * "0")) for g in STATE_GROUPS))

"""
    load_state_noise(path=STATE_NOISE_FILE; warn=true) -> StateNoise

The state's process noise and initial widths as the bench earned them (`estimator/bench_state_noise.py`
writes the file with the physics fingerprint it ran on); the `StateNoise` defaults, with a note, when there is
no file yet, and a warning when the tree's physics or filter no longer match the file's fingerprint (the
rates are then stale: rerun the bench).
"""
function load_state_noise(path::AbstractString=STATE_NOISE_FILE; warn::Bool=true)
    if !isfile(path)
        warn && @info "no state-noise file at $path: the StateNoise defaults, a starting point and not a result; earn them with estimator/bench_state_noise.py"
        return StateNoise()
    end
    d = JSON.parsefile(path)
    got = string(get(get(d, "physics", Dict()), "fingerprint", ""))
    fp = physics_fingerprint()
    got == fp || !warn || @warn "the state noise in $path was earned on physics $got; the tree's is $fp: the rates are stale, rerun estimator/bench_state_noise.py"
    kw = Dict{Symbol,Float64}()
    for (k, v) in d["rates"]; kw[Symbol(k)] = Float64(v); end
    for (k, v) in d["init_sd"]; kw[Symbol(k * "0")] = Float64(v); end
    return StateNoise(; kw...)
end

"""
    Past

The objective's backward-looking integrals, accumulated by the filter from its own solves as the cook goes, so a
plan from the filtered state never re-solves the past: the lethality integrals on the surface and the core
[60 °C-equivalent s], the bark-burn term's numerator and denominator (time integrals of the squared excess over
the cap while in the pit, and of the in-pit weight), the fuel loaded at the start [kg], and the time the
integrals run to [s]. `cook_outputs` adds them to a future solve's.
"""
mutable struct Past
    leth_s::Float64
    leth_c::Float64
    surf_num::Float64
    surf_den::Float64
    m0::Float64
    t::Float64
end
Past() = Past(0.0, 0.0, 0.0, 0.0, NaN, 0.0)
function _accumulate!(past::Past, o, tw, t_pull, g::PlanGoals)
    lg(x) = 1 / (1 + exp(-x))
    sp(x) = x > 30 ? x : log1p(exp(x))
    leth(T) = 10^((T - g.leth_t_ref) / g.leth_z)
    ip(tk) = 1 - lg((tk - t_pull) / DELTA_PULL_S)
    sq(T) = sp((T - g.t_surf_max) / g.dT_soft)^2
    for j in 1:(length(tw) - 1)
        dt = tw[j + 1] - tw[j]
        past.leth_s += dt * (leth(o.Tsurf[j]) + leth(o.Tsurf[j + 1])) / 2
        past.leth_c += dt * (leth(o.Tcore[j]) + leth(o.Tcore[j + 1])) / 2
        past.surf_num += dt * (ip(tw[j]) * sq(o.Tsurf[j]) + ip(tw[j + 1]) * sq(o.Tsurf[j + 1])) / 2
        past.surf_den += dt * (ip(tw[j]) + ip(tw[j + 1])) / 2
    end
    past.t = tw[end]
    return past
end

"""
    PhysicsState(area_latent, events, lay, first)

The slice's physics for the graph: the monolith solved on the slice's grid from the state in the block with
the latents in the block, the same stacked outputs the channels read (`_stack_1d`), and the end state with
its Jacobian kept from the last linearization (`last`) for the prediction to the next update, with the solve's
output for the past's integrals. `first`: the very first slice, where the ignited fraction is the latent a0
rather than a state. The discrepancies do not enter the physics: their columns are zero.
"""
struct PhysicsState <: Physics
    area_latent::Int
    events::NamedTuple
    lay::FilterLayout
    first::Bool
    last::Base.RefValue{Any}
end
PhysicsState(area_latent::Int, events::NamedTuple, lay::FilterLayout, first::Bool) = PhysicsState(area_latent, events, lay, first, Ref{Any}(nothing))
function _state_solve(v, d::GateData, ph::PhysicsState)
    lay = ph.lay
    z = v[1:lay.n_z]; x = v[state_range(lay)]
    xp = from_z(z[1:N_PHYS], d.prior[1:N_PHYS])
    θ, φ = xp[1:N_THETA], xp[(N_THETA + 1):N_PHYS]
    ics, a = unpack_state(x, lay)
    φs = vcat(φ[1:6], ph.first ? φ[7] : clamp(a, 1e-6, 1 - 1e-6))
    A = ph.area_latent > 0 ? exp(z[ph.area_latent]) : d.cfg.A_meat
    o = monolith_solve(θ, φs, d.u, d.t, d.t; pit=d.pit, cfg=d.cfg, abstol=d.abstol, reltol=d.reltol, A_meat=A, ics..., ph.events...)
    ics2, a2 = handoff(o, length(d.t))
    return _stack_1d(o, A, θ[4]), pack_state(ics2, a2, lay), OutputLayout(size(o.T_field, 1), length(d.t)), o
end
physics_stack(v, d::GateData, ph::PhysicsState) = (r = _state_solve(v, d, ph); (r[1], r[3]))
function physics_outputs(v, d::GateData, ph::PhysicsState)
    L = ph.last[]
    L !== nothing && L.c == v && return L.S, L.J, L.lay                 # the last linearization, at this point already
    lay = ph.lay
    nzx = lay.n_z + lay.n_state
    g(vv) = (r = _state_solve(vv, d, ph); vcat(r[1], r[2]))          # over the latents and the state: the discrepancies are not in the physics
    S, x_end, olay, o = _state_solve(v, d, ph)
    n_s = length(S)
    vzx = v[1:nzx]
    Jzx = ForwardDiff.jacobian(g, vzx, ForwardDiff.JacobianConfig(g, vzx, ForwardDiff.Chunk{12}()))
    J = hcat(Jzx, zeros(size(Jzx, 1), lay.n_ch))
    ph.last[] = (c=copy(v), S=S, J=J[1:n_s, :], lay=olay, x_end=x_end, J_end=J[(n_s + 1):end, :], o=o)
    return S, J[1:n_s, :], olay
end

"""
    FilterState

The filter's belief between updates and everything it carries: the layout and the joint sites; the mean and
covariance of the block (the state at the last update, the discrepancies there, the latents); the registry's
channels on the full grid (their data and masks the caller fills as samples arrive); per channel the noise prior
(`np0`, the calibration's) and the current posterior (`np`), the level, the fault flag and when the level went
high, the time of the last identification, the residual at the predicted mean at each of the channel's samples (the
reading less the physics) with the belief's own prediction variance there, the innovation standardized by its
predicted variance at each sample and the latest smoothed variance of those, and the grid indices; the state's process noise, the latents' (`q` per hour, `jumps`), the objective's constants and the
past's integrals; the grid index the belief is at and the update count; the cook's grid, damper and executed
events (the caller updates `u` and `events` as the cook goes); the physics' area latent and tolerances; the
fit's iterations and the level's settings.
"""
mutable struct FilterState
    lay::FilterLayout
    joint::Vector{PriorSite}
    sites::Vector{PriorSite}
    names::Vector{String}
    m::Vector{Float64}
    P::Matrix{Float64}
    channels::Vector{Channel}
    np0::Vector{NoisePrior}
    np::Vector{NoisePrior}
    level::Vector{Float64}
    faulty::Vector{Bool}
    high_since::Vector{Float64}
    t_ident::Vector{Float64}
    dser::Vector{Vector{Float64}}
    dvar::Vector{Vector{Float64}}
    eser::Vector{Vector{Float64}}
    vlev::Vector{Float64}
    didx::Vector{Vector{Int}}
    sn::StateNoise
    q::Vector{Float64}
    jumps::Vector{Tuple{Int,Float64,Float64}}
    goals::PlanGoals
    past::Past
    k0::Int
    n_updates::Int
    t::Vector{Float64}
    u::Vector{Float64}
    pit::PitConfig
    cfg::MeatConfig
    events::NamedTuple
    area_latent::Int
    abstol::Float64
    reltol::Float64
    iters::Int
    level_w::Float64
    level_window::Int
    fault_level::Float64
    ident_min::Int
    adapt_level::Bool
    ftol::Float64
end

"""
    filter_init(channels, t, u, pit, cfg, sites, names, events; noise, area_latent, sn=StateNoise(), q, jumps, goals, iters=1, ...) -> FilterState

The belief before any sample: the latents at the prior's centre with the prior's widths, the state the recipe's
(a zero-length solve at the prior centre, handed off at its first point) with the `StateNoise` initial widths,
every discrepancy at zero with its stationary spread. `channels` are the registry's channels on the grid `t`
(one `NoisePrior` each in `noise`, in order); `u` the damper on the grid (NaN where not yet executed, the caller
fills it); `q` the process-noise variance rate per hour per latent; `jumps` a vector of `(site, t_jump, var)`;
`iters` the most Gauss-Newton steps an update takes and `ftol` the objective drop under which it stops (a
routine update converges in one or two; a large innovation, the first scale reading against a prior area a
third off, needs the iteration: one step of an exponential overshoots and leaves a covariance that claims
otherwise);
`ident_min` the residuals a channel must have before its (σ, ρ) are re-identified (the prior until then);
`adapt_level` whether each channel's level follows its standardized innovations (off on the bench that earns
the state's process noise: a level reads the belief's consistency and cannot be judged before the belief is
consistent; the dead-reading flag stays on).
"""
function filter_init(channels::AbstractVector{<:Channel}, t, u, pit::PitConfig, cfg::MeatConfig, sites::Vector{PriorSite}, names::Vector{String},
                     events::NamedTuple; noise::AbstractVector{NoisePrior}, area_latent::Int, sn::StateNoise=StateNoise(),
                     q::AbstractVector=zeros(length(sites)), jumps::AbstractVector=[], goals::PlanGoals=PlanGoals(),
                     abstol::Float64=1e-6, reltol::Float64=1e-6, iters::Int=4, ftol::Float64=0.05, level_w::Float64=0.05, level_window::Int=100, fault_level::Float64=3.0,
                     ident_min::Int=24, adapt_level::Bool=true)
    n_z = length(sites); n_ch = length(channels)
    length(noise) == n_ch || throw(ArgumentError("one NoisePrior per channel ($(length(noise)) for $n_ch channels)"))
    length(names) == n_z || throw(ArgumentError("one name per site"))
    z0 = z_center.(sites)
    xp0 = from_z(z0[1:N_PHYS], sites[1:N_PHYS])
    A0 = area_latent > 0 ? exp(z0[area_latent]) : cfg.A_meat
    o0 = monolith_solve(xp0[1:N_THETA], xp0[(N_THETA + 1):N_PHYS], [isnan(u[1]) ? 0.15 : u[1], isnan(u[2]) ? 0.15 : u[2]], t[1:2], t[1:2];
                        pit, cfg, A_meat=A0, abstol, reltol, events...)
    ics0, a0 = handoff(o0, 1)
    lay = FilterLayout(length(ics0.T0f), n_z, n_ch)
    joint = vcat(sites, state_sites(lay.n_state), disc_sites(n_ch))
    m = vcat(z0, pack_state(ics0, a0, lay), zeros(n_ch))
    _, _, h0 = prior_terms(z0, sites)
    P = zeros(lay.n, lay.n)
    for i in 1:n_z; P[i, i] = 1 / h0[i]; end                                  # the latents: the prior's widths
    for (j, i) in enumerate(state_range(lay)); P[i, i] = state_init_sd(sn, lay)[j]^2; end   # the state: the recipe's, to its own uncertainty
    for c in 1:n_ch; i = disc_index(lay, c); P[i, i] = noise[c].σ_mean^2; end   # the discrepancies: their stationary spread
    past = Past(); past.m0 = o0.m_fuel[1]; past.t = t[1]
    return FilterState(lay, joint, collect(sites), collect(String, names), m, P, collect(Channel, channels), collect(NoisePrior, noise), collect(NoisePrior, noise),
                       ones(n_ch), falses(n_ch), fill(NaN, n_ch), fill(NaN, n_ch), [Float64[] for _ in 1:n_ch], [Float64[] for _ in 1:n_ch], [Float64[] for _ in 1:n_ch], ones(n_ch), [Int[] for _ in 1:n_ch],
                       sn, collect(Float64, q), [(Int(j[1]), Float64(j[2]), Float64(j[3])) for j in jumps], goals, past, 1, 0,
                       collect(Float64, t), u, pit, cfg, events, area_latent, abstol, reltol, iters, level_w, level_window, fault_level, ident_min, adapt_level, ftol)
end

# a channel's residual series (and the variances beside it) split into runs at gaps longer than three times
# its spacing, and the spacing
function _series_runs(vals::AbstractVector, vars::AbstractVector, idx::AbstractVector{Int})
    n = length(idx)
    n == 0 && return Vector{Float64}[], Vector{Float64}[], 1
    gaps = diff(idx)
    g = isempty(gaps) ? 1 : max(1, sort(gaps)[div(length(gaps) + 1, 2)])
    maxgap = g == 1 ? 1 : 3g
    runs = Vector{Float64}[]; vruns = Vector{Float64}[]
    cur = [Float64(vals[1])]; vcur = [Float64(vars[1])]
    for j in 2:n
        if idx[j] - idx[j - 1] <= maxgap
            push!(cur, vals[j]); push!(vcur, vars[j])
        else
            push!(runs, cur); push!(vruns, vcur); cur = [Float64(vals[j])]; vcur = [Float64(vars[j])]
        end
    end
    push!(runs, cur); push!(vruns, vcur)
    return runs, vruns, g
end

"A channel's correlation time [s] from its per-step correlation."
_corr_time(np::NoisePrior, dt) = -dt / log(clamp(np.ρ_mean, 1e-6, 0.999))
"The level that scales a channel's discrepancy innovation: its own, or the fault threshold's when the channel is flagged (a frozen sensor's low level would otherwise make the belief trust it more)."
_effective_level(st::FilterState, c::Int) = st.faulty[c] ? max(st.level[c], st.fault_level) : st.level[c]

# the noise of channel c after it delivered a sample at k: (σ, ρ) on the AR node over the full residual series
# (each residual observing the discrepancy through the belief's own prediction variance there) with the
# calibration's prior, when the series has grown by one correlation time since the last identification; the
# level from the volatility smoother on the channel's standardized innovations against the other channels', at
# every sample; the fault flag
function _identify!(st::FilterState, c::Int, k::Int)
    t = st.t; dt = t[2] - t[1]; tk = t[k]
    runs, vruns, g = _series_runs(st.dser[c], st.dvar[c], st.didx[c])
    isnan(st.t_ident[c]) && (st.t_ident[c] = t[st.didx[c][1]])
    τ = _corr_time(st.np[c], dt)
    # a correlation time is when the posterior of (σ, ρ) can have moved; the AR node still needs a series to read
    # them from (eight residuals of the humidity once put its ρ at 0.82 against 0.96 and made every reading look
    # four times as informative): at least `ident_min` samples, the calibration's prior until then
    if tk - st.t_ident[c] >= τ && length(st.dser[c]) >= st.ident_min && any(length(r) >= 3 for r in runs)
        keepr = [i for i in eachindex(runs) if length(runs[i]) >= 2]
        est = ar1_noise(runs[keepr], g, st.np0[c]; eps=vruns[keepr])
        st.np[c] = NoisePrior(est.σ, est.post.σ_sd, est.ρ, est.post.ρ_sd; floor=st.np0[c].floor)
        st.t_ident[c] = tk
    end
    # the level: the smoothed variance of the channel's standardized innovations over its trailing window (the
    # volatility smoother on a white unit series); the channel's level moves by its square root, so its
    # discrepancy innovation grows until its innovations are unit again (the adaptive filter's fixed point), and
    # never falls under half the identified noise. It reads the belief's consistency on that channel, so it is
    # judged only once the belief is consistent (`adapt_level`).
    e = st.eser[c]
    if st.adapt_level && length(e) >= 4
        win = e[max(1, length(e) - st.level_window + 1):end]
        vc = volatility_level(win, 0.0, 1.0; w=st.level_w).level[end]^2
        st.vlev[c] = isfinite(vc) ? clamp(vc, 1e-3, 1e4) : 1.0
        st.level[c] = clamp(st.level[c] * sqrt(st.vlev[c]), 0.5, 100.0)
    end
    # the flag: the smoother's level is already a judgement over the trailing window, so a level over the
    # threshold is a sensor whose readings carry more than the belief can explain (a burst, a jump, a reading
    # stuck at a wrong value). A reading frozen at its last value carries no noise at all, which innovation power
    # cannot see through a discrepancy as persistent as a probe's (its innovations are small either way): five
    # identical readings in a row are a dead sensor. Either way the belief stops trusting it: a flagged channel's
    # discrepancy innovation is inflated to the threshold's level (`_effective_level`), so its rows are discounted.
    y = rawdata(st.channels[c])
    n_s = length(st.didx[c])
    dead = n_s >= 5 && y !== nothing && all(y[st.didx[c][j]] == y[st.didx[c][end]] for j in (n_s - 4):n_s)
    if st.level[c] > st.fault_level || dead
        isnan(st.high_since[c]) && (st.high_since[c] = tk)
        st.faulty[c] = true
    elseif st.level[c] < 2 / 3 * st.fault_level
        st.high_since[c] = NaN
        st.faulty[c] = false
    end
    return st
end

"""
    filter_update!(st, k; diagnostics=false) -> record

The belief carried from its grid point to `k` and updated on the samples there: the discrepancies through their
AR(1) transitions over the interval, the physics solved on the interval from the state in the block, the graph
fit on the samples at `k` (skipped when none arrived: the belief is only propagated), the noise and the level of
each channel that reported, the state predicted to `k` through the physics with the transition's Jacobian and
the latents with their process noise, the past's integrals extended. The channels' samples strictly between the
last update and `k` are an error: the update happens at every arrival. Returns one record: `t`, the belief's
pit, centre and surface temperatures `(mean, sd)` and their joint covariance, the latents' posterior, the
discrepancies, the noise and level per channel, the fault flags, the fit's misfit, iterations and wall; with
`diagnostics`, the innovations at the predicted mean (before the update): `nis`, `dof` and the standardized
innovations per channel.
"""
function filter_update!(st::FilterState, k::Int; diagnostics::Bool=false)
    lay = st.lay; t = st.t; k0 = st.k0
    k > k0 || throw(ArgumentError("the belief is at grid point $k0; an update must be at a later point, not $k"))
    n_z, n_ch = lay.n_z, lay.n_ch
    for (c, ch) in enumerate(st.channels), j in (k0 + 1):(k - 1)
        ch.mask[j] && throw(ArgumentError("channel $c has a sample at grid point $j between the last update ($k0) and this one ($k): update at every arrival"))
    end
    for (c, ch) in enumerate(st.channels)
        (k0 == 1 && ch.mask[1] && rawdata(ch) !== nothing && !isfinite(rawdata(ch)[1])) && throw(ArgumentError("channel $c has a sample at the first grid point but no finite reading"))
    end
    idx = k0:k; tw = t[idx]; uw = st.u[idx]
    any(isnan, uw) && throw(ArgumentError("the damper is not set on the whole interval $k0:$k"))
    nsteps = k - k0
    t0 = time()
    # 1. the discrepancies to k: their AR(1) transition over the interval, the level in the innovation
    m, P = copy(st.m), copy(st.P)
    for c in 1:n_ch
        i = disc_index(lay, c)
        ρn = clamp(st.np[c].ρ_mean, 0.0, 0.999)^nsteps
        m[i] *= ρn
        P[i, :] .*= ρn; P[:, i] .*= ρn
        P[i, i] += st.np[c].σ_mean^2 * (1 - ρn^2) * _effective_level(st, c)^2
    end
    # 2. the interval's physics and channels
    d = estimator_setup(tw, uw, st.pit, st.cfg, st.sites[1:N_PHYS], st.names[1:N_PHYS]; abstol=st.abstol, reltol=st.reltol)
    ph = PhysicsState(st.area_latent, st.events, lay, k0 == 1)
    keep = falses(length(idx)); keep[end] = true                      # the interval's channels: the sample at k, the earlier points the last update's
    k0 == 1 && (keep[1] = true)                                       # the readings at the cook's first point join the first update (nothing came before them)
    for (c, ch) in enumerate(st.channels)
        ch.mask[k] && rawdata(ch) !== nothing && !isfinite(rawdata(ch)[k]) && throw(ArgumentError("channel $c has a sample at grid point $k but no finite reading"))
    end
    chs = Channel[Sequential(restrict(window(ch, idx), keep), disc_index(lay, c), white_floor(ch)) for (c, ch) in enumerate(st.channels)]
    live = [c for c in 1:n_ch if size(whitening(chs[c]), 1) > 0]
    prior_w = FilterPrior(st.joint, m, inv(Symmetric(P)))
    nis, dof, innov = NaN, 0, Dict{Int,Vector{Float64}}()
    resid = Dict{Int,Float64}(); rvar = Dict{Int,Float64}()
    if isempty(live)
        physics_outputs(m, d, ph)                                         # the transition's Jacobian at the predicted mean
        z_post, V_post, mis, n_it, converged = m, P, nothing, 0, true
    else
        # the residual of every sample at the predicted mean, before the update: the reading less the physics, the
        # series the noise and the level are identified on (the discrepancy as the instrument sees it, not as the
        # belief's own prior on it shrinks it); with `diagnostics` also the innovations whitened by their predicted
        # covariance (the linearization at the predicted mean is the solve's first, kept)
        Sp, Jp, layp = physics_outputs(m, d, ph)
        a, B, w = assemble(chs, m, Sp, Jp, prior_w, layp)
        row = 0
        for (c, ch) in enumerate(chs)
            nr = size(whitening(ch), 1)
            nr == 0 && continue
            if count(noise_mask(ch)) > 0                                  # a capped reading is not a residual
                σw = white_floor(ch)
                resid[c] = (w[row + 1] - a[row + 1]) * σw + m[disc_index(lay, c)]
                Bx = B[row + 1, :]; Bx[disc_index(lay, c)] = 0.0             # the belief's prediction variance of the physics there, the discrepancy's own excluded
                rvar[c] = σw^2 * (dot(Bx, P, Bx) + 1.0)
            end
            row += nr
        end
        # the innovations whitened by their predicted covariance (the belief's uncertainty of the physics, the
        # discrepancy's own spread and the floor): white, zero-mean, unit when the filter is consistent. The level
        # reads them; with `diagnostics` they are recorded with the normalized innovation squared
        keepr = findall(isfinite, w)
        if !isempty(keepr)
            nu = w[keepr] .- a[keepr]; H = B[keepr, :]
            Sm = Symmetric(H * P * H' + I)
            nis = dot(nu, Sm \ nu); dof = length(nu)
            sdev = sqrt.(max.(diag(Sm), 1e-12))
            pos = Dict(i => j for (j, i) in enumerate(keepr))
            row = 0
            for (c, ch) in enumerate(chs)
                nr = size(whitening(ch), 1)
                nr == 0 && continue
                innov[c] = [(w[i] - a[i]) / sdev[pos[i]] for i in (row + 1):(row + nr) if haskey(pos, i)]
                row += nr
            end
        end
        res = solve_map_channels(chs, prior_w, d; c0=m, physics=ph, iters=st.iters, ftol=st.ftol, verbose=false)
        z_post, V_post, mis, n_it, converged = res.z, Matrix(Symmetric(res.cov)), res.misfit, length(res.hist), res.converged
    end
    # 3. the residual series, the noise and the level of every channel that reported a reading
    for c in live
        haskey(resid, c) || continue
        push!(st.dser[c], resid[c]); push!(st.dvar[c], rvar[c]); push!(st.didx[c], k)
        push!(st.eser[c], haskey(innov, c) && !isempty(innov[c]) ? innov[c][1] : 0.0)
        _identify!(st, c, k)
    end
    # 4. the prediction of the state to k: the end state and its Jacobian at the MAP, the latents with process noise
    L = ph.last[]
    (L === nothing || maximum(abs.(L.c .- z_post)) > 1e-9) && (physics_outputs(z_post, d, ph); L = ph.last[])
    F = Matrix(1.0 * I, lay.n, lay.n)
    F[state_range(lay), :] = L.J_end
    Δh = (tw[end] - tw[1]) / 3600
    Q = zeros(lay.n, lay.n)
    for i in 1:n_z; Q[i, i] = st.q[i] * Δh; end
    for (s_i, t_j, var) in st.jumps
        (tw[1] < t_j <= tw[end]) && (Q[s_i, s_i] += var)
    end
    rates = state_rates(st.sn, lay)
    for (j, i) in enumerate(state_range(lay)); Q[i, i] = rates[j]^2 * Δh; end
    m_new = copy(z_post)
    m_new[state_range(lay)] = L.x_end
    m_new[ignited_index(lay)] = clamp(m_new[ignited_index(lay)], 1e-6, 1 - 1e-6)
    P_new = Matrix(Symmetric(F * V_post * F' + Q))
    st.m, st.P = m_new, P_new
    st.k0 = k; st.n_updates += 1
    _accumulate!(st.past, L.o, tw, haskey(st.events, :t_pull) ? st.events.t_pull : BrisketDynamics.PULL_NEVER, st.goals)
    diagnostics || (nis = NaN; dof = 0; innov = Dict{Int,Vector{Float64}}())
    sd = sqrt.(max.(diag(P_new), 0.0))
    iT = [pit_index(lay), centre_index(lay), surface_index(lay)]
    xr = state_range(lay)                                             # with diagnostics the whole state block and its covariance (the bench scores it by group)
    return (t=tw[end], k=k, updated=!isempty(live), channels=live,
            T_pit=(m_new[iT[1]], sd[iT[1]]), T_centre=(m_new[iT[2]], sd[iT[2]]), T_surface=(m_new[iT[3]], sd[iT[3]]), T_cov=P_new[iT, iT],
            x=(diagnostics ? m_new[xr] : nothing), x_cov=(diagnostics ? P_new[xr, xr] : nothing),
            z=m_new[1:n_z], z_sd=sd[1:n_z], z_cov=P_new[1:n_z, 1:n_z], disc=m_new[disc_range(lay)], disc_sd=sd[disc_range(lay)],
            noise=[(σ=np.σ_mean, σ_sd=np.σ_sd, ρ=np.ρ_mean, ρ_sd=np.ρ_sd) for np in st.np], level=copy(st.level), faulty=copy(st.faulty),
            misfit=mis, iters=n_it, converged=converged, wall=time() - t0, nis=nis, dof=dof, innov=innov)
end

"The first grid point after `k0` where any channel has a sample, or `k_to` when none does before it."
function _next_arrival(st::FilterState, k0::Int, k_to::Int)
    for k in (k0 + 1):k_to
        any(ch.mask[k] for ch in st.channels) && return k
    end
    return k_to
end

"""
    filter_advance!(st; t_to, diagnostics=false, verbose=false) -> Vector{record}

The belief carried to `t_to`: one update at every grid point where a channel delivers a sample, a propagation
to the grid point at or before `t_to` when none does; the records of `filter_update!`.
"""
function filter_advance!(st::FilterState; t_to::Float64, diagnostics::Bool=false, verbose::Bool=false)
    k_to = findlast(st.t .<= t_to + 1e-9)
    out = NamedTuple[]
    while st.k0 < k_to
        k = _next_arrival(st, st.k0, k_to)
        hour_before = floor(st.t[st.k0] / 3600 + 1e-9)
        r = filter_update!(st, k; diagnostics)
        push!(out, r)
        if verbose && floor(r.t / 3600 + 1e-9) > hour_before
            @printf("  t %5.2f h: pit %.1f ± %.1f, centre %.1f ± %.1f, surface %.1f ± %.1f C; levels %s; %d updates, %.1f s each\n",
                    r.t / 3600, r.T_pit[1] - 273.15, r.T_pit[2], r.T_centre[1] - 273.15, r.T_centre[2], r.T_surface[1] - 273.15, r.T_surface[2],
                    join([@sprintf("%.1f", l) for l in r.level], " "), st.n_updates, sum(x.wall for x in out) / length(out))
            flush(stdout)
        end
    end
    return out
end

"""
    sequential_filter(channels, t, u, pit, cfg, sites, names, events; noise, area_latent, q, jumps, sn, diagnostics, verbose, t_end, kw...) -> Vector{record}

The filter over a whole record: `filter_init` then `filter_advance!` to `t_end` (the grid's end by default).
"""
function sequential_filter(channels::AbstractVector{<:Channel}, t, u, pit::PitConfig, cfg::MeatConfig, sites::Vector{PriorSite}, names::Vector{String},
                           events::NamedTuple; noise::AbstractVector{NoisePrior}, area_latent::Int, q::AbstractVector=zeros(length(sites)),
                           jumps::AbstractVector=[], sn::StateNoise=StateNoise(), diagnostics::Bool=false, verbose::Bool=false, t_end::Float64=Inf, kw...)
    st = filter_init(channels, t, collect(Float64, u), pit, cfg, sites, names, events; noise, area_latent, sn, q, jumps, kw...)
    return filter_advance!(st; t_to=min(t_end, t[end]), diagnostics, verbose)
end

"""
    state_now(st) -> (k, t, ics, ignited, x, past, zx_cov)

The plant's state the belief holds at its grid point, for a plan from it: the grid index and time, the
initial-condition keywords and the ignited fraction (`handoff`'s contract), the packed state vector, the past's
integrals, and the joint covariance of the latents and the state (`plan_predictive` carries it into the plan).
"""
function state_now(st::FilterState)
    lay = st.lay
    x = st.m[state_range(lay)]
    ics, a = unpack_state(x, lay)
    nzx = lay.n_z + lay.n_state
    return (k=st.k0, t=st.t[st.k0], ics=ics, ignited=clamp(a, 1e-6, 1 - 1e-6), x=x, past=st.past, zx_cov=st.P[1:nzx, 1:nzx], lay=lay)
end

"""
    innovation_misfit(recs; window_s=3000.0, t_now) -> (z, nis, dof)

The misfit of the model over the trailing `window_s` seconds of updates (the pit's correlation time by default):
the pooled normalized innovation squared against its degrees of freedom as a z-score, (Σ NIS − Σ dof) / √(2 Σ
dof); positive when the innovations carry more power than the belief admits (the model is wrong somewhere), the
one side the replan trigger reads. `nothing` without diagnostics in the records.
"""
function innovation_misfit(recs::AbstractVector; window_s::Float64=3000.0, t_now::Float64=recs[end].t)
    s_nis = 0.0; s_dof = 0
    for r in recs
        (r.t > t_now - window_s - 1e-9 && r.dof > 0 && isfinite(r.nis)) || continue
        s_nis += r.nis; s_dof += r.dof
    end
    s_dof == 0 && return nothing
    return (z=(s_nis - s_dof) / sqrt(2 * s_dof), nis=s_nis, dof=s_dof)
end
