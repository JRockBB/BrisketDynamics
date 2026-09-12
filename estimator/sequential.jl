# --- the sequential filter on the graph: the belief over the state and the latents, slice by slice ---------
#
# filter.jl's shape, with the plant's state IN the belief. The latent block is [z (the seventeen latents); x
# (the plant's state at the slice's start: the meat's temperature, water, collagen, fat and salt fields, the
# bark and the ring, the pit's temperature, fuel, pan, firebox and ignited fraction, as `handoff` carries
# them)]. A slice is the same graph fit as every other estimator here: `solve_map_channels` on the graph
# engine, the same channel objects restricted to the slice's samples (one node per whitened residual, the
# censored humidity, the scale, the probe at a latent depth), the AR node identifying each channel's noise with
# the previous slice's posterior as its prior, the physics linearized and re-centred by the iterated loop.
# The prior on the block is a full-covariance Gaussian (FilterPrior): the previous slice's posterior carried
# through the physics to the slice's start, the transition's Jacobian in the block taken from the same
# ForwardDiff pass the graph linearizes with. The drift sites carry process noise at their rates and the add's
# charge a jump at the add, so a parameter that moves is followed; the state's uncertainty is in the belief,
# so no hand-off is taken as known, which is what filter.jl paid for on the leak and the drip.
#
# The belief at every slice, the pit, centre and surface temperatures with their widths, is what the pilot
# plans on and what a figure of the cook must draw.

"The layout of the joint block: the latents first (the channels' latent indices unchanged), then the state."
struct FilterLayout
    nx::Int              # meat nodes
    n_state::Int         # 5 nx fields + bark_b, bark_s, ring + Tpit, m_fuel, m_pan, T_pan, T_fb, ignited
    n_z::Int
    n::Int
end
FilterLayout(nx::Int, n_z::Int) = FilterLayout(nx, 5 * nx + 9, n_z, n_z + 5 * nx + 9)
const STATE_SCALARS = (:B0, :S0, :R0, :Tpit0, :mfuel0, :mpan0, :Tpan0, :Tfb0)     # after the five fields; then the ignited fraction
state_range(lay::FilterLayout) = (lay.n_z + 1):lay.n
"The block index of the pit's temperature, the centre node and the surface node of the temperature field (the probe position formula: depth 0 is node nx, the centre node 1)."
pit_index(lay::FilterLayout) = lay.n_z + 5 * lay.nx + 4
centre_index(lay::FilterLayout) = lay.n_z + 1
surface_index(lay::FilterLayout) = lay.n_z + lay.nx
"The state block's sites: Gaussian in the raw coordinate (`from_z` is the identity on the \"linear\" family)."
state_sites(n::Int) = [PriorSite("linear", 0.0, 1.0) for _ in 1:n]

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
    PhysicsState(area_latent, events, lay, first)

The slice's physics for the graph: the monolith solved on the slice's grid from the state in the block with
the latents in the block, the same stacked outputs the channels read (`_stack_1d`), and the end state with
its Jacobian kept from the last linearization (`last`) for the prediction to the next slice. `first`: the
very first slice, where the ignited fraction is the latent a0 rather than a state.
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
    return _stack_1d(o, A, θ[4]), pack_state(ics2, a2, lay), OutputLayout(size(o.T_field, 1), length(d.t))
end
physics_stack(v, d::GateData, ph::PhysicsState) = (r = _state_solve(v, d, ph); (r[1], r[3]))
function physics_outputs(v, d::GateData, ph::PhysicsState)
    g(vv) = (r = _state_solve(vv, d, ph); vcat(r[1], r[2]))
    S, x_end, lay = _state_solve(v, d, ph)
    n_s = length(S)
    J = ForwardDiff.jacobian(g, v, ForwardDiff.JacobianConfig(g, v, ForwardDiff.Chunk{12}()))
    ph.last[] = (c=copy(v), x_end=x_end, J_end=J[(n_s + 1):end, :])
    return S, J[1:n_s, :], lay
end

"""
    FilterState

The filter's belief between calls: the layout, the joint sites, the mean and covariance of the block at the
next slice's start, the channels' noise posteriors, the grid index of the next slice's start and the slice
count. `filter_init` makes it from the recipe and the prior; `filter_advance!` carries it to a time.
"""
mutable struct FilterState
    lay::FilterLayout
    joint::Vector{PriorSite}
    m::Vector{Float64}
    P::Matrix{Float64}
    np::Dict{Int,NoisePrior}
    k0::Int
    slice::Int
end

"""
    filter_init(t, u, pit, cfg, sites, events; noise, area_latent, x_jitter=1e-3) -> FilterState

The belief before any sample: the latents at the prior's mean with the prior's widths, the state the recipe's
(a zero-length solve at the prior mean, handed off at its first point) with a nominal jitter.
"""
function filter_init(t, u, pit::PitConfig, cfg::MeatConfig, sites::Vector{PriorSite}, events::NamedTuple;
                     noise::AbstractDict{Int,NoisePrior}, area_latent::Int, x_jitter::Float64=1e-3, abstol=1e-6, reltol=1e-6)
    n_z = length(sites); sites_phys = sites[1:N_PHYS]
    z0 = [isbeta(p) ? log(p.p1 / p.p2) : p.p1 for p in sites]
    xp0 = from_z(z0[1:N_PHYS], sites_phys)
    o0 = monolith_solve(xp0[1:N_THETA], xp0[(N_THETA + 1):N_PHYS], u[1:2], t[1:2], t[1:2]; pit, cfg, A_meat=exp(z0[area_latent]), abstol, reltol, events...)
    ics0, a0 = handoff(o0, 1)
    lay = FilterLayout(length(ics0.T0f), n_z)
    joint = vcat(sites, state_sites(lay.n_state))
    m = vcat(z0, pack_state(ics0, a0, lay))
    _, _, h0 = prior_terms(z0, sites)
    P = zeros(lay.n, lay.n)
    for i in 1:n_z; P[i, i] = 1 / h0[i]; end                          # the latents: the prior's widths
    for i in state_range(lay); P[i, i] = x_jitter^2; end               # the initial state is the recipe's: nearly known
    return FilterState(lay, joint, m, P, Dict{Int,NoisePrior}(i => v for (i, v) in noise), 1, 0)
end

"""
    filter_advance!(st, readings, masks, t, u, pit, cfg, sites, names, events; t_to, step, cap, depth_latent, area_latent,
                    q, jumps, x_jitter, iters, noise_every, diagnostics, verbose) -> Vector{NamedTuple}

The filter carried from its state to `t_to`, one slice of `step` seconds at a time (the last slice ends at the
grid point at or before `t_to`), under the events as they stand now; returns the slices' records (see
`sequential_filter`) and leaves `st` at the last slice's end.
"""
function filter_advance!(st::FilterState, readings, masks, t, u, pit::PitConfig, cfg::MeatConfig, sites::Vector{PriorSite}, names::Vector{String},
                         events::NamedTuple; t_to::Float64, step::Float64=600.0, cap::Float64, depth_latent::Int, area_latent::Int,
                         q::AbstractVector, jumps::AbstractVector=[], x_jitter::Float64=1e-3, iters::Int=2, noise_every::Int=6,
                         diagnostics::Bool=false, verbose::Bool=false)
    dt = t[2] - t[1]
    per = max(1, round(Int, step / dt))
    n_z = length(sites); sites_phys = sites[1:N_PHYS]
    lay, joint = st.lay, st.joint
    np = st.np
    out = NamedTuple[]
    k_to = findlast(t .<= t_to + 1e-9)
    while st.k0 < length(t) && st.k0 + per <= k_to
        st.slice += 1
        slice = st.slice
        k0 = st.k0
        m, P = st.m, st.P
        k1 = min(k0 + per, length(t)); idx = k0:k1

        tw, uw = t[idx], u[idx]
        t0 = time()
        # the slice's channels: the same objects as everywhere, the readings on the slice; its first point belongs
        # to the previous slice
        chs = [PitProbe(np[1].σ_mean, np[1].ρ_mean, Bool.(masks.pit[idx]); data=Float64.(readings.pit[idx])),
               MeatProbe(np[2].σ_mean, np[2].ρ_mean, Bool.(masks.core[idx]); depth_latent, data=Float64.(readings.core[idx])),
               CensoredHumidity(np[3].σ_mean, np[3].ρ_mean, Bool.(masks.humidity[idx]); cap, data=Float64.(readings.humidity[idx])),
               MassScale(np[4].σ_mean, np[4].ρ_mean, Bool.(masks.mass[idx]); data=Float64.(readings.mass[idx]))]
        for ch in chs; ch.mask[1] = false; end
        d = estimator_setup(tw, uw, pit, cfg, sites_phys, names)
        ph = PhysicsState(area_latent, events, lay, k0 == 1)
        prior_w = FilterPrior(joint, m, inv(Symmetric(P)))
        identify = slice % noise_every == 0
        # the innovations: every sample's whitened residual at the predicted mean, before the update
        nis, dof, innov = NaN, 0, Dict{Int,Vector{Float64}}()
        if diagnostics
            Sp, Jp, layp = physics_outputs(m, d, ph)
            a, B, w = assemble(chs, m, Sp, Jp, prior_w, layp)
            keepr = findall(isfinite, w)
            if !isempty(keepr)
                nu = w[keepr] .- a[keepr]; H = B[keepr, :]
                Sm = Symmetric(H * P * H' + I)
                nis = dot(nu, Sm \ nu); dof = length(nu)
                sdev = sqrt.(max.(diag(Sm), 1e-12))
                pos = Dict(i => j for (j, i) in enumerate(keepr))
                row = 0
                for (ci, ch) in enumerate(chs)
                    nr = size(whitening(ch), 1)
                    innov[ci] = [(w[i] - a[i]) / sdev[pos[i]] for i in (row + 1):(row + nr) if haskey(pos, i)]
                    row += nr
                end
            end
        end
        res = solve_map_channels(chs, prior_w, d; c0=m, physics=ph, engine=:graph, noise=identify ? np : nothing, noise_rounds=1,
                                 levelled=false, iters=identify ? max(iters, 3) : iters, verbose=false)
        for (i, v) in res.noise
            np[i] = NoisePrior(v.σ, v.σ_sd, v.ρ, v.ρ_sd; floor=np[i].floor)
        end
        # the prediction: the slice's end state and its Jacobian at the MAP, the latents carried with process noise
        L = ph.last[]
        (L === nothing || maximum(abs.(L.c .- res.z)) > 1e-9) && (physics_outputs(res.z, d, ph); L = ph.last[])
        F = vcat(hcat(Matrix(1.0 * I, n_z, n_z), zeros(n_z, lay.n_state)), L.J_end)
        Q = zeros(lay.n, lay.n)
        Δh = (tw[end] - tw[1]) / 3600
        for i in 1:n_z; Q[i, i] = q[i] * Δh; end
        for (s_i, t_j, var) in jumps
            (tw[1] < t_j <= tw[end]) && (Q[s_i, s_i] += var)
        end
        for i in state_range(lay); Q[i, i] = x_jitter^2; end
        Pz = Matrix(Symmetric(res.cov))
        m = vcat(res.z[1:n_z], L.x_end)
        P = Matrix(Symmetric(F * Pz * F' + Q))
        m[lay.n] = clamp(m[lay.n], 1e-6, 1 - 1e-6)
        st.m, st.P = m, P
        sd = sqrt.(max.(diag(P), 0.0))
        iT = [pit_index(lay), centre_index(lay), surface_index(lay)]
        push!(out, (t=tw[end], T_pit=(m[pit_index(lay)], sd[pit_index(lay)]), T_centre=(m[centre_index(lay)], sd[centre_index(lay)]),
                    T_surface=(m[surface_index(lay)], sd[surface_index(lay)]), T_cov=P[iT, iT], z=m[1:n_z], z_sd=sd[1:n_z], z_cov=P[1:n_z, 1:n_z],
                    noise=res.noise, misfit=res.misfit, iters=length(res.hist), converged=res.converged, wall=time() - t0,
                    nis=nis, dof=dof, innov=innov))
        verbose && @printf("  t %5.2f h: pit %.1f ± %.1f, centre %.1f ± %.1f, surface %.1f ± %.1f C; %d iterations, F %.2f; %.0f s\n",
                           tw[end] / 3600, m[pit_index(lay)] - 273.15, sd[pit_index(lay)], m[centre_index(lay)] - 273.15, sd[centre_index(lay)],
                           m[surface_index(lay)] - 273.15, sd[surface_index(lay)], length(res.hist), res.F, time() - t0)
        verbose && flush(stdout)
        st.k0 = k1
    end
    return out
end

"""
    sequential_filter(readings, masks, t, u, pit, cfg, sites, names, events; step, noise, cap, depth_latent, area_latent,
                      q, jumps, x_jitter, iters, noise_every, diagnostics, verbose, t_end) -> Vector{NamedTuple}

The filter over a whole record: `filter_init` then `filter_advance!` to `t_end`. Returns one record per slice:
`t` (the slice's end), the belief's pit, centre and surface temperatures `(mean, sd)` there and their joint
covariance `T_cov`, the latents' posterior `z`, `z_sd`, `z_cov`, the noise posteriors, the fit's misfit,
iterations and wall; with `diagnostics`, the innovations at the predicted mean, `nis`, `dof` and the
standardized innovations per channel (`innov`).
"""
function sequential_filter(readings, masks, t, u, pit::PitConfig, cfg::MeatConfig, sites::Vector{PriorSite}, names::Vector{String},
                           events::NamedTuple; step::Float64=600.0, noise::AbstractDict{Int,NoisePrior}, cap::Float64,
                           depth_latent::Int, area_latent::Int, q::AbstractVector, jumps::AbstractVector=[],
                           x_jitter::Float64=1e-3, iters::Int=2, noise_every::Int=6, diagnostics::Bool=false, verbose::Bool=false, t_end::Float64=Inf)
    st = filter_init(t, u, pit, cfg, sites, events; noise, area_latent, x_jitter)
    return filter_advance!(st, readings, masks, t, u, pit, cfg, sites, names, events; t_to=min(t_end, t[end]), step, cap, depth_latent,
                           area_latent, q, jumps, x_jitter, iters, noise_every, diagnostics, verbose)
end
