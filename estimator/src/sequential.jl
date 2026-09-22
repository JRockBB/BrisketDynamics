# --- the filter on the graph: the belief over the plant's state, the latents and the instruments' discrepancies,
# --- updated as each measurement arrives ----------------------------------------------------------------------
#
# The latent block is [z (the latents); x (unconstrained softplus/logit coordinates for the plant's positive and
# unit-interval state at the current update: the meat's temperature, water, collagen, fat and salt fields, the
# bark and the ring, the pit's temperature, fuel, pan, firebox and ignited fraction, decoded to `handoff`'s
# physical contract); d (one AR(1) discrepancy per channel: the coloured part of its error, the
# calibration's noise model as a state)]. The filter first predicts that block to grid point k: moving latents
# receive their process noise before the interval's physics, the state is propagated through the physics, state
# and event disturbances are added at the endpoint, and discrepancies take their AR(1) transitions. The endpoint
# belief is then the full-covariance Gaussian prior (`FilterPrior`) of the same RxInfer graph fit as every other
# estimator here: `solve_map_channels` over the registry's samples at k wrapped as `Sequential` (the reading is
# the endpoint physics plus the channel's discrepancy plus white noise at the instrument's floor), one node per
# residual and the censored humidity's survival-likelihood site. Thus the reading at k conditions every uncertainty that
# accumulated before it. Where no sample arrives the belief is only propagated. Nothing here runs on a clock:
# the update happens when a sample arrives; a channel's noise (σ, ρ)
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
fuel_index(lay::FilterLayout) = lay.n_z + 5 * lay.nx + 5
firebox_index(lay::FilterLayout) = lay.n_z + 5 * lay.nx + 8
"The state block's sites: Gaussian in an unconstrained coordinate; `pack_state`/`unpack_state` enforce its physical domain."
state_sites(n::Int) = [PriorSite("linear", 0.0, 1.0) for _ in 1:n]
"The discrepancy sites: the same linear family, in the instrument's units."
disc_sites(n::Int) = state_sites(n)

const STATE_POSITIVE_FLOOR = 1e-8
const STATE_UNIT_EPS = 1e-8
_state_softplus(y) = y > 30 ? y : log1p(exp(y))
_state_invsoftplus(x) = x > 30 ? x : log(expm1(x))

"The coordinate kind of every state entry: positive softplus coordinate or unit-interval logit coordinate."
function state_kinds(lay::FilterLayout)
    nx = lay.nx
    return vcat(fill(:positive, 2nx), fill(:unit, nx), fill(:positive, 2nx),
                [:positive, :positive, :unit, :positive, :positive, :positive, :positive, :positive, :unit])
end

function _state_encode(x, kind::Symbol)
    if kind == :positive
        floor = oftype(x, STATE_POSITIVE_FLOOR)
        return _state_invsoftplus(max(x - floor, floor * eps(Float64)))
    end
    ε = oftype(x, STATE_UNIT_EPS)
    q = clamp((x - ε) / (one(x) - 2ε), eps(Float64), 1 - eps(Float64))
    return log(q / (one(q) - q))
end
_state_decode(y, kind::Symbol) =
    kind == :positive ? oftype(y, STATE_POSITIVE_FLOOR) + _state_softplus(y) :
    oftype(y, STATE_UNIT_EPS) + (one(y) - 2oftype(y, STATE_UNIT_EPS)) * logistic(y)

"Encode a physical state in the Gaussian filter coordinates."
state_to_coord(x, lay::FilterLayout) = [_state_encode(v, k) for (v, k) in zip(x, state_kinds(lay))]
"Decode the Gaussian filter coordinates to a state that is strictly in its physical domain."
state_from_coord(y, lay::FilterLayout) = [_state_decode(v, k) for (v, k) in zip(y, state_kinds(lay))]
"The diagonal of ∂ physical state / ∂ filter coordinate."
function state_decode_jacobian(y, lay::FilterLayout)
    return [k == :positive ? logistic(v) :
            (one(v) - 2oftype(v, STATE_UNIT_EPS)) * logistic(v) * (one(v) - logistic(v))
            for (v, k) in zip(y, state_kinds(lay))]
end

"Map a physical covariance into local filter coordinates, limiting one interval's coordinate sd for boundary stability."
function _state_cov_to_coord(Q, y, lay::FilterLayout; max_sd::Float64=3.0)
    dxdy = state_decode_jacobian(y, lay)
    d = inv.(max.(dxdy, eps(Float64)))
    Qy = Matrix(Symmetric((d .* Q) .* d'))
    sd = sqrt.(max.(diag(Qy), 0.0))
    scale = min.(1.0, max_sd ./ max.(sd, eps(Float64)))
    return Matrix(Symmetric((scale .* Qy) .* scale'))
end

"Decode a state Gaussian by first-order moment propagation, for physical diagnostics."
function state_physical_moments(y, P, lay::FilterLayout)
    x = state_from_coord(y, lay)
    d = state_decode_jacobian(y, lay)
    return x, Matrix(Symmetric((d .* P) .* d'))
end

"The filter-coordinate state vector from `handoff`'s physical initial conditions and ignited fraction."
function pack_state(ics::NamedTuple, a, lay::FilterLayout)
    x = vcat(ics.T0f, ics.W0f, ics.C0f, ics.F0f, ics.Ssalt0f,
             [getfield(ics, k) for k in STATE_SCALARS], [a])
    return state_to_coord(x, lay)
end
"The physical initial-condition keywords and ignited fraction decoded from a filter-coordinate state vector."
function unpack_state(y, lay::FilterLayout)
    x = state_from_coord(y, lay)
    nx = lay.nx
    f(i) = x[((i - 1) * nx + 1):(i * nx)]
    sc = Dict(k => x[5 * nx + j] for (j, k) in enumerate(STATE_SCALARS))
    ics = (T0f=f(1), W0f=f(2), C0f=f(3), F0f=f(4), Ssalt0f=f(5), B0=sc[:B0], S0=sc[:S0], R0=sc[:R0],
           Tpit0=sc[:Tpit0], mfuel0=sc[:mfuel0], mpan0=sc[:mpan0], Tpan0=sc[:Tpan0], Tfb0=sc[:Tfb0])
    return ics, x[5 * nx + 9]
end

"""
    StateNoise(; T, W, C, F, S, bark, T_pit, m_fuel, m_pan, T_pan, T_fb, ignited, T0, W0, ...)

The state's process noise and initial widths by field group, in physical units: a rate per √hour for
the temperature field [K], the water [kg/kg], the collagen, fat and salt fields, the bark and ring scalars, the
pit's temperature [K], fuel [kg], pan mass [kg] and temperature [K], the firebox [K] and the ignited fraction;
and the sd of each at the start (the recipe's meat and pit known to about a kelvin, the loads to a few
percent). Each covariance is mapped locally into the state's constrained coordinate, with any one interval's
coordinate sd capped at three to remain numerically meaningful next to a physical boundary. The rates are what
the consistency bench earns (NEES of the temperatures and of the latents in their bands); the defaults are a
starting point, not a result, and a coordinate change requires re-earning them.

`T_event`, `T_spritz` and `m_add` are the disturbances an event leaves on the endpoint state: the interval that
crosses a lid-open event (the add, the wrap, the pull, a spritz) adds their variance to the pit's and the
firebox's temperatures, and the add's to the fuel mass. They enter the predicted belief before the endpoint
reading, so that reading can condition the disturbed state rather than forcing the event's innovation onto a
constant physical parameter.

`fields` says how a field's noise is spread over its nodes. `:offset` (the default) moves every node of the
field together: the recipe's meat is uniform to a kelvin as a whole, and a model's error in a diffusion field is
smooth, so the field's uncertainty is its bulk state and its shape stays the physics'. `:nodes` gives every node
its own independent width. Independent nodes let an update bend the field between two adjacent nodes to fit a
probe's own noise, and a probe whose depth is a latent then reads the bend as depth information it never had
(the whitened depth sensitivity at a first update went from 0 at the prior to 73 at the optimum on the bench's
seed 7); water diffusion is far too slow to smooth per-node noise between updates, so a water field's surface
node detaches from the bulk. A tiny per-node jitter (`node_jitter` of the group's width) keeps the covariance
positive definite under `:offset`.
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
    fields::Symbol = :offset
    node_jitter::Float64 = 0.003
    T_event::Float64 = 5.0            # the pit's and the firebox's temperatures after a lid-open event the belief propagated through (the add, the wrap, the pull) [K]
    T_spritz::Float64 = 1.5           # the same after a spritz [K]
    m_add::Float64 = 0.15             # the fuel mass after the add: the chimney's mass is not the config's to the gram [kg]
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

"""
    state_noise_cov(lay, sn, sd::AbstractVector) -> Matrix

The covariance the per-entry widths `sd` (from `state_rates` scaled by √Δh, or `state_init_sd`) give the state
block under the noise's structure: under `:offset` every field's nodes share one draw (the block is sd²·11ᵀ plus
the jitter on its diagonal), the scalars are independent; under `:nodes` everything is diagonal.
"""
function state_noise_cov(lay::FilterLayout, sn::StateNoise, sd::AbstractVector)
    Q = Matrix(Diagonal(Float64.(sd) .^ 2))
    sn.fields == :nodes && return Q
    sn.fields == :offset || throw(ArgumentError("StateNoise.fields must be :offset or :nodes, not $(sn.fields)"))
    for (g, rg) in state_groups(lay)
        length(rg) > 1 && g in ("T", "W", "C", "F", "S") || continue
        s = sd[first(rg)]
        Q[rg, rg] .= s^2
        for i in rg; Q[i, i] = s^2 * (1 + sn.node_jitter^2); end
    end
    return Q
end
"The degrees of freedom a group's consistency is scored on: one per node under `:nodes`, one (the bulk) for a field under `:offset`."
group_dofs(lay::FilterLayout, sn::StateNoise) = Dict(g => (sn.fields == :offset && length(rg) > 1 && g in ("T", "W", "C", "F", "S") ? 1 : length(rg)) for (g, rg) in state_groups(lay))

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

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
"The state-noise file the bench writes and every driver reads."
const STATE_NOISE_FILE = joinpath(REPO_ROOT, "data", "calibration", "state_noise.json")
"The filter's own sources, fingerprinted with the physics: a change to any of them means the rates must be re-earned."
const FILTER_FILES = ("BrisketEstimator.jl", "sequential.jl", "channels.jl", "noise.jl", "volatility.jl", "realcook.jl", "drift.jl", "segments.jl", "iterated.jl")

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
    append!(files, ["estimator/src/" * f for f in FILTER_FILES])
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
                                        "init_sd" => Dict(g => getfield(sn, Symbol(g * "0")) for g in STATE_GROUPS),
                                        "fields" => string(sn.fields), "node_jitter" => sn.node_jitter,
                                        "events" => Dict("T_event" => sn.T_event, "T_spritz" => sn.T_spritz, "m_add" => sn.m_add))

"""
    load_state_noise(path=STATE_NOISE_FILE; warn=true) -> StateNoise

The state's process noise and initial widths as the bench earned them (`estimator/python/bench_state_noise.py`
writes the file with the physics fingerprint it ran on); the `StateNoise` defaults, with a note, when there is
no file yet, and a warning when the tree's physics or filter no longer match the file's fingerprint (the
rates are then stale: rerun the bench).
"""
function load_state_noise(path::AbstractString=STATE_NOISE_FILE; warn::Bool=true)
    if !isfile(path)
        warn && @info "no state-noise file at $path: the StateNoise defaults, a starting point and not a result; earn them with estimator/python/bench_state_noise.py"
        return StateNoise()
    end
    d = JSON.parsefile(path)
    got = string(get(get(d, "physics", Dict()), "fingerprint", ""))
    fp = physics_fingerprint()
    got == fp || !warn || @warn "the state noise in $path was earned on physics $got; the tree's is $fp: the rates are stale, rerun estimator/python/bench_state_noise.py"
    kw = Dict{Symbol,Any}()
    for (k, v) in d["rates"]; kw[Symbol(k)] = Float64(v); end
    for (k, v) in d["init_sd"]; kw[Symbol(k * "0")] = Float64(v); end
    haskey(d, "fields") && (kw[:fields] = Symbol(d["fields"]))
    haskey(d, "node_jitter") && (kw[:node_jitter] = Float64(d["node_jitter"]))
    for (k, v) in get(d, "events", Dict()); kw[Symbol(k)] = Float64(v); end
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
    sigma::Matrix{Float64}            # the prediction's and the update's sigma design (N × n_z); empty: the tangent throughout
    prior_cov::Base.RefValue{Any}     # the belief's covariance the update's points are placed by (the predicted prior at this update)
    pts::Base.RefValue{Any}           # the points at the last centre: the centre, the design, the latent steps, the stacks, the joint vectors
    Jx::Base.RefValue{Any}            # the tangent computed once per update at its first centre: its state columns serve every later centre
    scale::Base.RefValue{Any}         # the assembled rows' raw scales after the residual inflation (the diagnostics read them)
    omega::Base.RefValue{Any}         # per channel, the regression's residual variance per reading at the last linearization (the trials read it)
end
PhysicsState(area_latent::Int, events::NamedTuple, lay::FilterLayout, first::Bool) =
    PhysicsState(area_latent, events, lay, first, Ref{Any}(nothing), zeros(0, lay.n_z), Ref{Any}(nothing), Ref{Any}(nothing), Ref{Any}(nothing), Ref{Any}(nothing), Ref{Any}(nothing))
"The physics with the sigma design `sigma` and the belief's covariance `P` its update points are placed by."
PhysicsState(area_latent::Int, events::NamedTuple, lay::FilterLayout, first::Bool, sigma::AbstractMatrix, P::AbstractMatrix) =
    PhysicsState(area_latent, events, lay, first, Ref{Any}(nothing), Matrix{Float64}(sigma), Ref{Any}(Matrix(P)), Ref{Any}(nothing), Ref{Any}(nothing), Ref{Any}(nothing), Ref{Any}(nothing))
"Whether the physics linearizes statistically over sigma points (a design is set) rather than by the tangent."
has_sigma(ph::PhysicsState) = size(ph.sigma, 1) > 0

"""
    _sigma_points(v, d, ph) -> (c, E, L, S, V, lay)

The update's sigma points at the centre `v`: the latents stepped by the whitened design scaled by the Cholesky
factor of the belief's latent block (`ph.prior_cov`), the state and the discrepancies held at the centre's (the
points read the physics' partial response to the latents; the state's and the discrepancies' columns are the
tangent's and the identity, so the regression must not carry their conditional coupling twice), each solved
through the slice: the stacks the channels read and the joint vectors the channels' own latents are read from.
A failed or non-finite solve is an error naming the point.
"""
function _sigma_points(v, d::GateData, ph::PhysicsState)
    lay = ph.lay; n_z = lay.n_z
    P = ph.prior_cov[]
    P === nothing && error("the physics carries a sigma design but no covariance to place the points by")
    Vzz = Symmetric(Matrix(P[1:n_z, 1:n_z]))
    F = cholesky(Vzz; check=false)
    issuccess(F) || error("the belief's latent block is not positive definite at the update (smallest eigenvalue $(minimum(eigvals(Vzz)))): no sigma points can be placed")
    N = size(ph.sigma, 1)
    S = Vector{Vector{Float64}}(undef, N); V = Vector{Vector{Float64}}(undef, N)
    olay = nothing
    for i in 1:N
        δ = F.L * ph.sigma[i, :]
        vi = copy(v); vi[1:n_z] = v[1:n_z] + δ
        r = try
            _state_solve(vi, d, ph)
        catch err
            err isa InterruptException && rethrow()
            error("the physics failed at sigma point $i of $N of the update (the latents moved by $(round(norm(δ), digits=3)) in z, at most $(round(maximum(abs.(δ)), digits=3)) on one site): $(sprint(showerror, err))")
        end
        all(isfinite, r[1]) || error("the physics returned a non-finite stack at sigma point $i of $N of the update (the latents moved by $(round(norm(δ), digits=3)) in z)")
        S[i] = r[1]; V[i] = vi; olay = r[3]
    end
    return (c=copy(v), E=ph.sigma, L=F.L, S=S, V=V, lay=olay)
end

"""
    assemble_at(channels, z, S, J, prior, lay, ph::PhysicsState) -> (a, B, w)

The affine node data by statistical linearization when the physics carries sigma points: for every channel its
readings at each point, regressed on the whitened latent steps, give the nodes' latent columns (the regression
slope, exact for a linear reading and the least-squares line through the belief's spread otherwise) and the
predicted reading at the centre becomes the points' mean; the state's and the discrepancies' columns are the
tangent's. The regression's residual variance on each row, the part of the reading's spread over the points no
line explains, inflates that row's noise: where the physics has a knee inside the belief's spread the reading
is trusted less, and the belief cannot be talked into a regime by the slope at its own mean. Without a design
this is `assemble`.
"""
function assemble_at(channels, z, S, J, prior, lay, ph::PhysicsState)
    has_sigma(ph) || return assemble(channels, z, S, J, prior, lay)
    pts = ph.pts[]
    pts === nothing && error("the sigma points have not been solved at this centre: the physics is read before the channels are assembled")
    pts.c == z || error("the sigma points were solved at another centre than the one assembled")
    n = length(z); n_z = ph.lay.n_z; N = length(pts.S)
    Jzero = zeros(length(pts.S[1]), n)
    as, Bs, ws, scales = Vector{Float64}[], Matrix{Float64}[], Vector{Float64}[], Vector{Float64}[]
    omegas = Dict{Int,Vector{Float64}}()
    for (ci, ch) in enumerate(channels)
        Lw = whitening(ch)
        size(Lw, 1) == 0 && continue
        r, Jr = predict(ch, S, J, z, prior, lay)
        R = hcat([predict(ch, pts.S[i], Jzero, pts.V[i], prior, pts.lay)[1] for i in 1:N]...)
        r̄ = vec(sum(R, dims=2) ./ N)
        Rc = R .- r̄
        G = (Rc * pts.E) ./ N                                            # the regression of the readings on the whitened steps
        Jr[:, 1:n_z] = G / pts.L                                         # the slope on the latents: G L⁻¹
        Ω = max.(vec(sum(Rc .^ 2, dims=2) ./ N) .- vec(sum(G .^ 2, dims=2)), 0.0)   # the residual the line leaves, per reading
        omegas[ci] = Ω
        sc = zeros(size(Lw, 1))
        for j in 1:size(Lw, 1)
            col = argmax(abs.(Lw[j, :]))
            s2 = 1 / Lw[j, col]^2                                        # the row's own variance in the reading's units
            Lw[j, :] .*= sqrt(s2 / (s2 + Ω[col]))
            sc[j] = sqrt(s2 + Ω[col])
        end
        y = rawdata(ch)
        a, B, w = _affine_rows(ch, r̄, Jr, Lw, y)
        push!(as, a); push!(Bs, B); push!(ws, w); push!(scales, sc)
    end
    ph.scale[] = reduce(vcat, scales)
    ph.omega[] = omegas
    return reduce(vcat, as), reduce(vcat, Bs), reduce(vcat, ws)
end

"""
    assemble_trial(channels, z, S, prior, lay, ph::PhysicsState) -> (a, w)

The local-Gaussian assembly at a trial centre under sigma points: the plain
slice's readings at the centre, each row inflated by the residual of the
last linearization (`ph.omega`). The nonlinear objective itself is evaluated
by `likelihood_objective`, including exact censored-normal survival terms.
Without a design this is `assemble`'s.
"""
function assemble_trial(channels, z, S, prior, lay, ph::PhysicsState)
    has_sigma(ph) || return (r = assemble(channels, z, S, zeros(length(S), length(z)), prior, lay); (r[1], r[3]))
    omegas = ph.omega[]
    omegas === nothing && error("a trial centre before any linearization: the rows' inflation is not known yet")
    Jzero = zeros(length(S), length(z))
    as, ws = Vector{Float64}[], Vector{Float64}[]
    for (ci, ch) in enumerate(channels)
        Lw = whitening(ch)
        size(Lw, 1) == 0 && continue
        r, _ = predict(ch, S, Jzero, z, prior, lay)
        Ω = omegas[ci]
        for j in 1:size(Lw, 1)
            col = argmax(abs.(Lw[j, :]))
            s2 = 1 / Lw[j, col]^2
            Lw[j, :] .*= sqrt(s2 / (s2 + Ω[col]))
        end
        y = rawdata(ch)
        a, _, w = _affine_rows(ch, r, Jzero, Lw, y)
        push!(as, a); push!(ws, w)
    end
    return reduce(vcat, as), reduce(vcat, ws)
end

function likelihood_objective(channels, z, S, prior, lay, ph::PhysicsState)
    sigma = has_sigma(ph)
    omegas = sigma ? ph.omega[] : nothing
    sigma && omegas === nothing &&
        error("a trial centre before any statistical linearization: the rows' inflation is not known yet")
    Jzero = zeros(length(S), length(z))
    loss = 0.0
    for (ci, ch) in enumerate(channels)
        Lw = whitening(ch)
        size(Lw, 1) == 0 && continue
        if sigma
            Ω = omegas[ci]
            for j in 1:size(Lw, 1)
                col = argmax(abs.(Lw[j, :]))
                s2 = 1 / Lw[j, col]^2
                Lw[j, :] .*= sqrt(s2 / (s2 + Ω[col]))
            end
        end
        r, _ = predict(ch, S, Jzero, z, prior, lay)
        loss += _channel_nll(ch, r, Lw, rawdata(ch))
    end
    return loss
end

function _state_solve(v, d::GateData, ph::PhysicsState)
    lay = ph.lay
    z = v[1:lay.n_z]; x = v[state_range(lay)]
    xp = from_z(z[1:N_PHYS], d.prior[1:N_PHYS])
    θ, φ = xp[1:N_THETA], xp[(N_THETA + 1):N_PHYS]
    ics, a = unpack_state(x, lay)
    φs = vcat(φ[1:6], ph.first ? φ[7] : a)
    A = ph.area_latent > 0 ? exp(z[ph.area_latent]) : d.cfg.A_meat
    o = monolith_solve(θ, φs, d.u, d.t, d.t; pit=d.pit, cfg=d.cfg, abstol=d.abstol, reltol=d.reltol, A_meat=A, ics..., ph.events...)
    ics2, a2 = handoff(o, length(d.t))
    return _stack_1d(o, A, θ[4], d.cfg), pack_state(ics2, a2, lay), OutputLayout(size(o.T_field, 1), length(d.t)), o
end
physics_stack(v, d::GateData, ph::PhysicsState) = (r = _state_solve(v, d, ph); (r[1], r[3]))    # a trial centre: the slice alone; the points belong to a linearization
"The tangent of the slice at `v`: the stacked outputs, their Jacobian over the latents and the state, the end state and its Jacobian, the solve."
function _tangent(v, d::GateData, ph::PhysicsState)
    lay = ph.lay
    nzx = lay.n_z + lay.n_state
    g(vv) = (r = _state_solve(vv, d, ph); vcat(r[1], r[2]))          # over the latents and the state: the discrepancies are not in the physics
    S, x_end, olay, o = _state_solve(v, d, ph)
    n_s = length(S)
    vzx = v[1:nzx]
    Jzx = ForwardDiff.jacobian(g, vzx, ForwardDiff.JacobianConfig(g, vzx, ForwardDiff.Chunk{12}()))
    J = hcat(Jzx, zeros(size(Jzx, 1), lay.n_ch))
    return (S=S, J=J[1:n_s, :], lay=olay, x_end=x_end, J_end=J[(n_s + 1):end, :], o=o)
end
function physics_outputs(v, d::GateData, ph::PhysicsState)
    L = ph.last[]
    L !== nothing && L.c == v && return L.S, L.J, L.lay                 # the last linearization, at this point already
    if has_sigma(ph)
        # the tangent once per update, at its first centre: its state columns serve every centre of the update
        # (the state's response over two minutes barely moves with the centre); the latent columns are the
        # regression's, at every centre; the points and the slice at this centre
        ph.Jx[] === nothing && (ph.Jx[] = _tangent(v, d, ph))
        T = ph.Jx[]
        S, x_end, olay, o = _state_solve(v, d, ph)
        ph.pts[] = _sigma_points(v, d, ph)
        ph.last[] = (c=copy(v), S=S, J=T.J, lay=olay, x_end=x_end, J_end=T.J_end, o=o, d=d)
        return S, T.J, olay
    end
    T = _tangent(v, d, ph)
    ph.last[] = (c=copy(v), S=T.S, J=T.J, lay=T.lay, x_end=T.x_end, J_end=T.J_end, o=T.o, d=d)
    return T.S, T.J, T.lay
end

"""
    sigma_design(n, N, seed) -> Matrix (N × n)

A fixed set of `N` points in `n` standard dimensions for the prediction's sigma points: `N ÷ 2` Gaussian draws
and their negatives (so the set's mean is exactly zero), whitened so the set's covariance is exactly the
identity. Any linear map of the belief is then reproduced exactly by the points, and a nonlinear one is sampled
at typical distances: in seventeen dimensions the textbook axis-aligned points sit four sd out along one axis
each and would probe the physics in regimes the belief does not hold, where these draws spread the same second
moment over every axis at about one sd apiece.
"""
function sigma_design(n::Int, N::Int, seed::Int)
    M = N ÷ 2
    M >= n || throw(ArgumentError("the sigma design needs at least 2n points ($N for n = $n)"))
    E0 = randn(MersenneTwister(seed), M, n)
    E = vcat(E0, -E0)
    C = Symmetric(E' * E ./ size(E, 1))
    Lc = cholesky(C).L
    return E / Lc'                                                    # rows ε̃ = L⁻¹ ε: the sample covariance is the identity
end

"""
    _sigma_prediction(lay, sigma, v, V, J_end, d, ph) -> (m_new, P_new)

The belief propagated to the slice's end through the physics by sigma points over the latents: each point
perturbs the latents by a whitened draw scaled by the posterior's Cholesky factor, takes the state and the
discrepancies at their conditional mean given that draw, and solves the slice; the latents' block is kept, the
cross-covariance and the state's covariance are the points' spread, and the state's own conditional uncertainty
(given the latents) rides the transition Jacobian at the optimum, the discrepancies theirs an identity. For a
linear slice this is the linear prediction exactly; for the cook's physics it carries the state's spread across
the regimes the latents' uncertainty reaches (a diffusivity one sd either way dries or wets the surface), which
the tangent at the mean cannot. A point whose solve fails, or a latents' block that is not positive definite,
is an error naming the point: there is no quiet way down to the tangent.
"""
function _sigma_prediction(lay::FilterLayout, sigma::AbstractMatrix, v::AbstractVector, V::AbstractMatrix, J_end::AbstractMatrix, d::GateData, ph::PhysicsState)
    n_z = lay.n_z; zr = 1:n_z; xr = state_range(lay); ξr = (n_z + 1):lay.n
    nx, nξ, N = length(xr), length(ξr), size(sigma, 1)
    Vzz = Symmetric(Matrix(V[zr, zr]))
    F = cholesky(Vzz; check=false)
    issuccess(F) || error("the latents' posterior covariance is not positive definite at the prediction (smallest eigenvalue $(minimum(eigvals(Vzz)))): no sigma points can be drawn from it")
    K = (F \ Matrix(V[zr, ξr]))'                                      # solve for the regression; do not form an inverse
    Vcond = _stable_cov(Matrix(V[ξr, ξr]) - K * Matrix(V[zr, ξr]))    # conditional spread, checked and repaired to PSD
    Δ = zeros(n_z, N); Ξ = zeros(nξ, N)
    for i in 1:N
        δ = F.L * sigma[i, :]
        ξ = v[ξr] + K * δ
        vi = copy(v); vi[zr] = v[zr] + δ; vi[xr] = ξ[1:nx]
        x_end = try
            _state_solve(vi, d, ph)[2]
        catch err
            err isa InterruptException && rethrow()
            error("the physics failed at sigma point $i of $N (the latents moved by $(round(norm(δ), digits=3)) in z, at most $(round(maximum(abs.(δ)), digits=3)) on one site): $(sprint(showerror, err))")
        end
        all(isfinite, x_end) || error("the physics returned a non-finite state at sigma point $i of $N (the latents moved by $(round(norm(δ), digits=3)) in z)")
        Δ[:, i] = δ; Ξ[:, i] = vcat(x_end, ξ[(nx + 1):end])
    end
    ξ̄ = vec(sum(Ξ, dims=2) ./ N)
    R = Ξ .- ξ̄
    A = zeros(nξ, nξ); A[1:nx, 1:nx] = J_end[:, xr]; for i in (nx + 1):nξ; A[i, i] = 1.0; end
    P = zeros(lay.n, lay.n)
    P[zr, zr] = Vzz
    P[zr, ξr] = (Δ * R') ./ N; P[ξr, zr] = P[zr, ξr]'
    P[ξr, ξr] = (R * R') ./ N + A * Vcond * A'
    m = copy(v); m[ξr] = ξ̄
    return m, Matrix(Symmetric(P))
end

"""
    FilterState

The filter's belief between updates and everything it carries: the layout and the joint sites; the mean and
covariance of the block (the constrained-coordinate state at the last update, the discrepancies there, the latents); the registry's
channels on the full grid (their data and masks the caller fills as samples arrive); per channel the noise prior
(`np0`, the calibration's) and the current posterior (`np`), the level, the fault flag and when the level went
high, the time of the last identification, the residual at the predicted mean at each of the channel's samples (the
reading less the physics) with the belief's own prediction variance there, the innovation standardized by its
predicted variance at each sample and the latest smoothed variance of those, and the grid indices; the state's process noise, the latents' (`q` per hour, `jumps`), the objective's constants and the
past's integrals; the grid index the belief is at and the update count; the cook's grid, damper and executed
events (the caller updates `u` and `events` as the cook goes); the physics' area latent and tolerances; the
fit's iterations and the level's settings; the prediction's sigma design (`sigma`, empty for the tangent
prediction).
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
    retry_iters::Int
    level_w::Float64
    level_window::Int
    fault_level::Float64
    ident_min::Int
    adapt_level::Bool
    ftol::Float64
    sigma::Matrix{Float64}
end

"""
    filter_init(channels, t, u, pit, cfg, sites, names, events; noise, area_latent, sn=StateNoise(), q, jumps, goals, iters=4, retry_iters=12, ...) -> FilterState

The belief before any sample: the latents at the prior's centre with the prior's widths, the constrained-coordinate
state of the recipe (a zero-length solve at the prior centre, handed off at its first point) with the physical
`StateNoise` initial widths mapped into that coordinate,
every discrepancy at zero with its stationary spread. `channels` are the registry's channels on the grid `t`
(one `NoisePrior` each in `noise`, in order); `u` the damper on the grid (NaN where not yet executed, the caller
fills it); `q` the process-noise variance rate per hour per latent; `jumps` a vector of `(site, t_jump, var)`;
`iters` the short first pass's Gauss-Newton steps, `retry_iters` the continuation from its last accepted center
when that pass does not converge (zero disables retry), and `ftol` the objective drop under which it stops;
`ident_min` the residuals a channel must have before its (σ, ρ) are re-identified (the prior until then) and before its level is judged (one until then);
`adapt_level` whether each channel's level follows its standardized innovations (off on the bench that earns
the state's process noise: a level reads the belief's consistency and cannot be judged before the belief is
consistent; the dead-reading flag stays on); `n_sigma` the sigma points of the prediction (`sigma_design`;
0 for the tangent prediction, the transition Jacobian at the optimum alone) and `sigma_seed` their draw.
"""
function filter_init(channels::AbstractVector{<:Channel}, t, u, pit::PitConfig, cfg::MeatConfig, sites::Vector{PriorSite}, names::Vector{String},
                     events::NamedTuple; noise::AbstractVector{NoisePrior}, area_latent::Int, sn::StateNoise=StateNoise(),
                     q::AbstractVector=zeros(length(sites)), jumps::AbstractVector=[], goals::PlanGoals=PlanGoals(),
                     abstol::Float64=1e-6, reltol::Float64=1e-6, iters::Int=4, retry_iters::Int=12,
                     ftol::Float64=0.05, level_w::Float64=0.05, level_window::Int=100, fault_level::Float64=3.0,
                     ident_min::Int=24, adapt_level::Bool=true, n_sigma::Int=64, sigma_seed::Int=7)
    n_z = length(sites); n_ch = length(channels)
    length(noise) == n_ch || throw(ArgumentError("one NoisePrior per channel ($(length(noise)) for $n_ch channels)"))
    length(names) == n_z || throw(ArgumentError("one name per site"))
    iters > 0 || throw(ArgumentError("iters must be positive"))
    retry_iters >= 0 || throw(ArgumentError("retry_iters must be nonnegative"))
    z0 = [z_moments(p)[1] for p in sites]                                  # the belief's mean: each family's mean in its coordinate
    xp0 = from_z(z0[1:N_PHYS], sites[1:N_PHYS])
    A0 = area_latent > 0 ? exp(z0[area_latent]) : cfg.A_meat
    o0 = monolith_solve(xp0[1:N_THETA], xp0[(N_THETA + 1):N_PHYS], [isnan(u[1]) ? 0.15 : u[1], isnan(u[2]) ? 0.15 : u[2]], t[1:2], t[1:2];
                        pit, cfg, A_meat=A0, abstol, reltol, events...)
    ics0, a0 = handoff(o0, 1)
    lay = FilterLayout(length(ics0.T0f), n_z, n_ch)
    joint = vcat(sites, state_sites(lay.n_state), disc_sites(n_ch))
    m = vcat(z0, pack_state(ics0, a0, lay), zeros(n_ch))
    P = zeros(lay.n, lay.n)
    for i in 1:n_z; P[i, i] = z_moments(sites[i])[2]; end                     # the latents: each family's variance in its coordinate
    yr = state_range(lay)
    Q0 = state_noise_cov(lay, sn, state_init_sd(sn, lay))
    P[yr, yr] = _state_cov_to_coord(Q0, m[yr], lay)                            # physical initial widths mapped into constrained state coordinates
    for c in 1:n_ch; i = disc_index(lay, c); P[i, i] = noise[c].σ_mean^2; end   # the discrepancies: their stationary spread
    past = Past(); past.m0 = o0.m_fuel[1]; past.t = t[1]
    return FilterState(lay, joint, collect(sites), collect(String, names), m, P, collect(Channel, channels), collect(NoisePrior, noise), collect(NoisePrior, noise),
                       ones(n_ch), falses(n_ch), fill(NaN, n_ch), fill(NaN, n_ch), [Float64[] for _ in 1:n_ch], [Float64[] for _ in 1:n_ch], [Float64[] for _ in 1:n_ch], ones(n_ch), [Int[] for _ in 1:n_ch],
                       sn, collect(Float64, q), [(Int(j[1]), Float64(j[2]), Float64(j[3])) for j in jumps], goals, past, 1, 0,
                       collect(Float64, t), u, pit, cfg, events, area_latent, abstol, reltol, iters, retry_iters,
                       level_w, level_window, fault_level, ident_min, adapt_level, ftol,
                       n_sigma > 0 ? sigma_design(n_z, n_sigma, sigma_seed) : zeros(0, n_z))
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

"A covariance with its marginal variances preserved and roundoff-scale negative correlation modes repaired."
function _stable_cov(P::AbstractMatrix; corr_floor::Float64=1e-12)
    S = Matrix(Symmetric(P))
    all(isfinite, S) || throw(ArgumentError("covariance contains a non-finite entry"))
    dscale = max(maximum(abs, diag(S)), 1.0)
    minimum(diag(S)) >= -1e-10 * dscale ||
        throw(ArgumentError("covariance has a materially negative marginal variance $(minimum(diag(S)))"))
    d = sqrt.(max.(diag(S), eps(Float64)))
    C = Symmetric(S ./ (d * d'))
    E = eigen(C)
    C1 = E.vectors * Diagonal(max.(E.values, corr_floor)) * E.vectors'
    dc = sqrt.(max.(diag(C1), eps(Float64)))
    C1 ./= dc * dc'                                                     # keep the original marginal variances
    return Matrix(Symmetric(C1 .* (d * d')))
end

"A rectangular square root of a positive-semidefinite covariance."
function _psd_factor(Q::AbstractMatrix)
    E = eigen(Symmetric(Matrix(Q)))
    vmax = max(maximum(E.values), 0.0)
    keep = findall(>(max(vmax, 1.0) * 1e-14), E.values)
    isempty(keep) && return zeros(size(Q, 1), 0)
    return E.vectors[:, keep] * Diagonal(sqrt.(max.(E.values[keep], 0.0)))
end

"Square-root covariance propagation, combining `F*P*F'` and independent `Q` by QR."
function _sqrt_propagate(F::AbstractMatrix, P::AbstractMatrix, Q::AbstractMatrix)
    Lp = cholesky(Symmetric(_stable_cov(P))).L
    A = hcat(F * Lp, _psd_factor(Q))
    R = Matrix(qr(A').R)
    return _stable_cov(R' * R)
end
_sqrt_cov_sum(P::AbstractMatrix, Q::AbstractMatrix) =
    _sqrt_propagate(Matrix{Float64}(I, size(P, 1), size(P, 1)), P, Q)

"A filter prior carrying the covariance's lower square root; the graph update runs in its whitened coordinates."
function _filter_prior(sites, m, P)
    Ps = _stable_cov(P)
    F = cholesky(Symmetric(Ps))
    n = size(Ps, 1)
    return FilterPrior(sites, collect(m), zeros(n, n), Matrix(F.L)), Ps
end

"Run an online RxInfer update, continuing once from its last accepted center when the short first pass does not converge."
function _online_solve(st::FilterState, chs, prior, d, ph, c0)
    res = solve_map_channels(chs, prior, d; c0, physics=ph, iters=st.iters, ftol=st.ftol, verbose=false)
    n_it = length(res.hist)
    if !res.converged && st.retry_iters > 0
        res = solve_map_channels(chs, prior, d; c0=res.z, physics=ph, iters=st.retry_iters, ftol=st.ftol, verbose=false)
        n_it += length(res.hist)
    end
    return res, n_it
end

# The measurement operator at one endpoint. A two-point, negligible-duration solve reconstructs every channel's
# instantaneous observable from the endpoint state (including humidity and bark algebra) while the channel masks
# retain only its first point. This keeps the ordinary channel registry and its RxInfer graph unchanged.
function _observation_context(st::FilterState, k::Int, P::AbstractMatrix; first::Bool=false)
    lay = st.lay
    δt = max(st.t[2] - st.t[1], 1.0) * 1e-6
    tobs = [st.t[k], st.t[k] + δt]
    uobs = [st.u[k], st.u[k]]
    d = estimator_setup(tobs, uobs, st.pit, st.cfg, st.sites[1:N_PHYS], st.names[1:N_PHYS]; abstol=st.abstol, reltol=st.reltol)
    ph = size(st.sigma, 1) > 0 ? PhysicsState(st.area_latent, st.events, lay, first, st.sigma, P) :
                                PhysicsState(st.area_latent, st.events, lay, first)
    idx = [k, k]
    keep = Bool[true, false]
    chs = Channel[Sequential(restrict(window(ch, idx), keep), disc_index(lay, c), white_floor(ch))
                  for (c, ch) in enumerate(st.channels)]
    live = [c for c in 1:lay.n_ch if size(whitening(chs[c]), 1) > 0]
    return d, ph, chs, live
end

"The physical process covariance accumulated over an interval, mapped into the endpoint's constrained coordinates."
function _state_process_cov(st::FilterState, tw, y_endpoint)
    lay = st.lay
    Δh = (tw[end] - tw[1]) / 3600
    Q = state_noise_cov(lay, st.sn, state_rates(st.sn, lay) .* sqrt(Δh))
    for (t_e, kind) in _event_times(st.events, tw[end])
        (tw[1] < t_e <= tw[end]) || continue
        σ = kind == :spritz ? st.sn.T_spritz : st.sn.T_event
        ip = pit_index(lay) - lay.n_z
        ifb = firebox_index(lay) - lay.n_z
        Q[ip, ip] += σ^2
        Q[ifb, ifb] += σ^2
        if kind == :add
            ifuel = fuel_index(lay) - lay.n_z
            Q[ifuel, ifuel] += st.sn.m_add^2
        end
    end
    return _state_cov_to_coord(Q, y_endpoint, lay)
end

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
    # the level: the smoothed variance, over the channel's trailing window, of its innovations standardized
    # WITHOUT the level (the volatility smoother on a white unit series); the level is that variance's square
    # root, a direct reading of how much more the channel's readings carry than the belief at level one can
    # explain, never under half the identified noise. A level that multiplied itself by the window's variance
    # compounded: two early outliers standardized under level one stayed in the window and drove the scale's
    # level to 100 while its later innovations were already unit. It is judged on no fewer samples than the
    # noise is, and only once the belief is consistent (`adapt_level`).
    e = st.eser[c]
    if st.adapt_level && length(e) >= st.ident_min
        win = e[max(1, length(e) - st.level_window + 1):end]
        vc = volatility_level(win, 0.0, 1.0; w=st.level_w).level[end]^2
        st.vlev[c] = isfinite(vc) ? clamp(vc, 1e-3, 1e4) : 1.0
        st.level[c] = clamp(sqrt(st.vlev[c]), 0.5, 100.0)
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

The belief predicted from its grid point to `k` and then updated on the samples there: an initial reading is
assimilated once before the first transition; moving latents receive their interval process noise; discrepancies
take their AR(1) transitions; the state is propagated through the interval physics; physical and event
disturbances enter the predicted endpoint covariance; and the graph fits only the endpoint samples (skipped when
none arrived). The noise and level of each reporting channel are then updated and the past's integrals extended.
The channels' samples strictly between the last update and `k` are an error: the update happens at every arrival.
If the short graph fit does not converge it is continued once from its last accepted center. If that also fails,
its center, covariance, residual history, and noise adaptation are discarded; the filter advances with the
predicted endpoint belief and reports `attempted=true, updated=false`.
Returns one record: `t`, the belief's
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
    m, P = copy(st.m), copy(st.P)

    # 1. The readings at the cook's initial point belong to the initial state, not to the discrepancy state at
    # the end of the first interval. Assimilate them once before the first prediction.
    if k0 == 1
        P = _stable_cov(P)
        d0, ph0, chs0, live0 = _observation_context(st, 1, P; first=true)
        if !isempty(live0)
            prior0, P = _filter_prior(st.joint, m, P)
            res0, n0 = _online_solve(st, chs0, prior0, d0, ph0, m)
            if res0.converged
                m, P = res0.z, _stable_cov(res0.cov)
            else
                @warn "initial observation update did not converge; retaining the initial belief" grid_index=1 iterations=n0
            end
        end
    end

    # 2. Predict to k. A moving latent's new interval value drives this interval's physics, so its process noise
    # enters before propagation. The discrepancy takes its independent AR transition.
    Δh = (tw[end] - tw[1]) / 3600
    for i in 1:n_z
        P[i, i] += st.q[i] * Δh
    end
    for (s_i, t_j, var) in st.jumps
        (tw[1] < t_j <= tw[end]) && (P[s_i, s_i] += var)
    end
    for c in 1:n_ch
        i = disc_index(lay, c)
        ρn = clamp(st.np[c].ρ_mean, 0.0, 0.999)^nsteps
        m[i] *= ρn
        P[i, :] .*= ρn; P[:, i] .*= ρn
        P[i, i] += st.np[c].σ_mean^2 * (1 - ρn^2) * _effective_level(st, c)^2
    end
    d_transition = estimator_setup(tw, uw, st.pit, st.cfg, st.sites[1:N_PHYS], st.names[1:N_PHYS]; abstol=st.abstol, reltol=st.reltol)
    ph_transition = PhysicsState(st.area_latent, st.events, lay, k0 == 1)
    L = _tangent(m, d_transition, ph_transition)
    sigma_ok = size(st.sigma, 1) > 0
    xr = state_range(lay)
    Q = zeros(lay.n, lay.n)
    Q[xr, xr] = _state_process_cov(st, tw, L.x_end)
    if sigma_ok
        m_pred, P_pred = _sigma_prediction(lay, st.sigma, m, P, L.J_end, d_transition, ph_transition)
        P_pred = _sqrt_cov_sum(P_pred, Q)
    else
        F = Matrix(1.0 * I, lay.n, lay.n)
        F[xr, :] = L.J_end
        m_pred = copy(m)
        m_pred[xr] = L.x_end
        P_pred = _sqrt_propagate(F, P, Q)
    end
    P_pred = _stable_cov(P_pred)

    # 3. Update the predicted endpoint belief with only the readings at k. Process and event uncertainty is now
    # in this prior, so those readings condition it directly; no event-specific measurement inflation is needed.
    for (c, ch) in enumerate(st.channels)
        ch.mask[k] && rawdata(ch) !== nothing && !isfinite(rawdata(ch)[k]) && throw(ArgumentError("channel $c has a sample at grid point $k but no finite reading"))
    end
    d, ph, chs, live = _observation_context(st, k, P_pred; first=false)
    nis, dof, innov = NaN, 0, Dict{Int,Vector{Float64}}()
    resid = Dict{Int,Float64}(); rvar = Dict{Int,Float64}(); efree = Dict{Int,Float64}()
    if isempty(live)
        z_post, V_post, mis, n_it, converged = m_pred, P_pred, nothing, 0, true
    else
        prior_w, P_pred = _filter_prior(st.joint, m_pred, P_pred)
        # the residual of every sample at the predicted mean, before the update: the reading less the physics, the
        # series the noise and the level are identified on (the discrepancy as the instrument sees it, not as the
        # belief's own prior on it shrinks it); with `diagnostics` also the innovations whitened by their predicted
        # covariance (the linearization at the predicted mean is the solve's first, kept)
        Sp, Jp, layp = physics_outputs(m_pred, d, ph)
        a, B, w = assemble_at(chs, m_pred, Sp, Jp, prior_w, layp, ph)
        rowscale = ph.scale[]                                             # under sigma points a row's scale carries the regression's residual
        row = 0
        for (c, ch) in enumerate(chs)
            nr = size(whitening(ch), 1)
            nr == 0 && continue
            if count(noise_mask(ch)) > 0                                  # a capped reading is not a residual
                σw = rowscale === nothing ? white_floor(ch) : rowscale[row + 1]
                resid[c] = (w[row + 1] - a[row + 1]) * σw + m_pred[disc_index(lay, c)]
                Bx = B[row + 1, :]; Bx[disc_index(lay, c)] = 0.0             # the belief's prediction variance of the physics there, the discrepancy's own excluded
                rvar[c] = σw^2 * (dot(Bx, P_pred, Bx) + 1.0)
            end
            row += nr
        end
        # the innovations whitened by their predicted covariance (the belief's uncertainty of the physics, the
        # discrepancy's own spread and the floor): white, zero-mean, unit when the filter is consistent. The level
        # reads them; with `diagnostics` they are recorded with the normalized innovation squared
        keepr = Int[]
        row = 0
        for ch in chs
            nr = size(whitening(ch), 1)
            append!(keepr, (row + j for j in 1:misfit_rows(ch) if isfinite(w[row + j])))
            row += nr
        end
        if !isempty(keepr)
            nu = w[keepr] .- a[keepr]; H = B[keepr, :]
            Sm = Symmetric(H * P_pred * H' + I)
            nis = dot(nu, Sm \ nu); dof = length(nu)
            sdev = sqrt.(max.(diag(Sm), 1e-12))
            pos = Dict(i => j for (j, i) in enumerate(keepr))
            row = 0
            for (c, ch) in enumerate(chs)
                nr = size(whitening(ch), 1)
                nr == 0 && continue
                innov[c] = [(w[i] - a[i]) / sdev[pos[i]] for i in (row + 1):(row + nr) if haskey(pos, i)]
                # the same innovation standardized as if the level were one: the discrepancy's innovation over the
                # interval carried (level² − 1) times its own variance, in the row's units, that the level reads
                if haskey(pos, row + 1)
                    ρn = clamp(st.np[c].ρ_mean, 0.0, 0.999)^nsteps
                    σrow = rowscale === nothing ? white_floor(ch) : rowscale[row + 1]
                    extra = (_effective_level(st, c)^2 - 1) * st.np[c].σ_mean^2 * (1 - ρn^2) / σrow^2
                    efree[c] = (w[row + 1] - a[row + 1]) / sqrt(max(sdev[pos[row + 1]]^2 - extra, 1e-6))
                end
                row += nr
            end
        end
        res, n_it = _online_solve(st, chs, prior_w, d, ph, m_pred)
        mis, converged = res.misfit, res.converged
        if converged
            z_post, V_post = res.z, _stable_cov(res.cov)
        else
            @warn "endpoint observation update did not converge; retaining the predicted belief" grid_index=k time=tw[end] iterations=n_it
            z_post, V_post = m_pred, P_pred
        end
    end
    # 4. the residual series, the noise and the level of every channel that reported a reading
    if converged
        for c in live
            haskey(resid, c) || continue
            push!(st.dser[c], resid[c]); push!(st.dvar[c], rvar[c]); push!(st.didx[c], k)
            push!(st.eser[c], get(efree, c, 0.0))
            _identify!(st, c, k)
        end
    end
    m_new = copy(z_post)
    P_new = _stable_cov(V_post)
    st.m, st.P = m_new, P_new
    st.k0 = k; st.n_updates += 1
    _accumulate!(st.past, L.o, tw, haskey(st.events, :t_pull) ? st.events.t_pull : BrisketDynamics.PULL_NEVER, st.goals)
    diagnostics || (nis = NaN; dof = 0; innov = Dict{Int,Vector{Float64}}())
    sd = sqrt.(max.(diag(P_new), 0.0))
    iT = [pit_index(lay), centre_index(lay), surface_index(lay)]
    xr = state_range(lay)                                             # with diagnostics the constrained-coordinate state block and covariance
    x_phys, P_phys = state_physical_moments(m_new[xr], P_new[xr, xr], lay)
    iTx = iT .- lay.n_z
    Tsd = sqrt.(max.(diag(P_phys[iTx, iTx]), 0.0))
    return (t=tw[end], k=k, attempted=!isempty(live), updated=!isempty(live) && converged, channels=live,
            T_pit=(x_phys[iTx[1]], Tsd[1]), T_centre=(x_phys[iTx[2]], Tsd[2]), T_surface=(x_phys[iTx[3]], Tsd[3]), T_cov=P_phys[iTx, iTx],
            x=(diagnostics ? m_new[xr] : nothing), x_cov=(diagnostics ? P_new[xr, xr] : nothing),
            z=m_new[1:n_z], z_sd=sd[1:n_z], z_cov=P_new[1:n_z, 1:n_z], disc=m_new[disc_range(lay)], disc_sd=sd[disc_range(lay)],
            noise=[(σ=np.σ_mean, σ_sd=np.σ_sd, ρ=np.ρ_mean, ρ_sd=np.ρ_sd) for np in st.np], level=copy(st.level), faulty=copy(st.faulty),
            misfit=mis, iters=n_it, converged=converged, wall=time() - t0, nis=nis, dof=dof, innov=innov, sigma=sigma_ok)
end

"The lid-open events of the cook up to `t_end`, each with its kind: the add, the wrap, the pull, and the spritz train to the pull."
function _event_times(events::NamedTuple, t_end::Float64)
    out = Tuple{Float64,Symbol}[]
    set(k) = haskey(events, k) && getfield(events, k) < 1e8
    set(:t_add) && push!(out, (events.t_add, :add))
    set(:t_wrap) && push!(out, (events.t_wrap, :wrap))
    set(:t_pull) && push!(out, (events.t_pull, :pull))
    if set(:t_spritz)
        tp = set(:t_pull) ? events.t_pull : t_end
        ts = events.t_spritz
        while ts <= min(tp, t_end)
            push!(out, (ts, :spritz))
            ts += events.dt_spritz
        end
    end
    return out
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
physical initial-condition keywords and ignited fraction (`handoff`'s contract), the packed constrained-coordinate
state vector, the past's integrals, and the joint covariance of the latents and that coordinate state
(`plan_predictive` carries it into the plan).
"""
function state_now(st::FilterState)
    lay = st.lay
    x = st.m[state_range(lay)]
    ics, a = unpack_state(x, lay)
    nzx = lay.n_z + lay.n_state
    return (k=st.k0, t=st.t[st.k0], ics=ics, ignited=a, x=x, past=st.past, zx_cov=st.P[1:nzx, 1:nzx], lay=lay)
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
