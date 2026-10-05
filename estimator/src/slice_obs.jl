# Readings observed through the physics from their slice's state node. A reading at t_r attaches to the newest state
# node at or before it (t0); its prediction is the physics carried from that node to t_r. Every reading of one slice
# shares ONE solve saving at all their times (and at the next node's, once the slice is closed) and one forward-mode
# Jacobian, so a sensor's rate adds rows, never solves, Jacobians or state variables. The state's process noise over
# the slice and each sensor's OU discrepancy enter as ONE joint Gaussian over the slice's readings, conditioned exactly
# on the slice's end values once a next node closes it:
#
#   x(t_i)  = Φ_i(y0, z) + a_i (y1 − Φ_end) + w_i,     w a random walk at the process rates, bridged to the end node
#   d_c(t_i) = α_i d_c(t0) + β_i d_c(t1) + e_i,          the OU conditional on the channel's values at the two nodes
#   v_i     = h_c(x(t_i), z) + d_c(t_i) + white_i
#
# The noise w, e and white are marginalized in the factor's covariance Σ (H Qk Hᵀ + the OU conditional + the white
# floors), fixed when the factor is built (a relinearization never changes the objective). The discrepancy has a
# variable only at state nodes, so the graph's size is set by the nodes, not by the readings.

"One uncensored reading of a slice: the sensor, value, time, compiled observable and the instrument's floor."
struct SliceReading
    spec::SensorSpec
    value::Float64
    t::Float64
    h::Function
    bias::Float64
    σ::Float64
end
function SliceReading(model::SystemModel, spec::SensorSpec, rec::MeasurementRecord)
    h = compile_observable(spec.observable, model.sys, model.bl, model.blk, model.log)
    bias, σ = 0.0, spec.noise.white_sd
    if model.rom !== nothing
        key = _rom_key(spec.observable)
        if key !== nothing
            e = _rom_observation_error(model.rom, spec.observable, key, rec.t)
            bias = e.bias; σ = sqrt(σ^2 + e.sd^2)
        end
    end
    return SliceReading(spec, rec.value, rec.t, h, bias, σ)
end

"""
    SliceObservationFactor

The readings of one slice (see the file's header): the latents (`key_z`, with `key_m` the moving copy at the anchor), the
anchor node `key_y0` at `t0`, the next node `key_y1` at `t1` once the slice is closed (`nothing`, `NaN` while open), per
reading its channel's discrepancy variables at the two nodes, the fixed bridge coefficients and the whitener of Σ. The
physics solve and its Jacobian are cached by point like a dynamics factor's.
"""
mutable struct SliceObservationFactor <: Factor
    model::SystemModel
    key_z::VarKey
    key_m::Union{Nothing,VarKey}
    split::Union{Nothing,LatentSplit}
    key_y0::VarKey
    key_y1::Union{Nothing,VarKey}
    t0::Float64
    t1::Float64
    first::Bool
    readings::Vector{SliceReading}
    kd0::Vector{Union{Nothing,VarKey}}
    kd1::Vector{Union{Nothing,VarKey}}
    a::Vector{Float64}             # the process bridge's weight on the end correction per reading (0 while open)
    αd::Vector{Float64}            # the OU conditional's weights on the channel's discrepancy at t0 and at t1
    βd::Vector{Float64}
    G::Matrix{Float64}             # per reading ∂h/∂x at its first prediction (physical units), fixed with the reading
    Σ::Matrix{Float64}             # the readings' joint noise covariance (fixed)
    Lw::Matrix{Float64}            # its whitener, inv(chol(Σ).L)
    keys::Vector{VarKey}           # the factor's variables in order: latents (static[, moving]), y0[, y1], the distinct discrepancies
    times::Vector{Float64}         # the solve's save times after t0: the distinct reading times, then t1 when closed
    chunk::Int
    relin_tol::Float64
    relin_tol_state::Float64
    cache::Any
    n_solves::Int
    n_jacobians::Int
end
vars(f::SliceObservationFactor) = f.keys
_search_residual_curvature(::SliceObservationFactor) = true
ndim(f::SliceObservationFactor) = length(f.readings)
provenance(f::SliceObservationFactor) = "$(length(f.readings)) readings through the physics from $(f.t0) s" * (f.key_y1 === nothing ? " (open)" : " to $(f.t1) s")

# stationary OU covariance between two times
_ou_cov(nm::NoiseModel, s, t) = nm.ar_sd^2 * exp(-abs(s - t) / nm.tau)

"""
    SliceObservationFactor(model, readings, key_z, key_y0, t0, first; key_y1, t1, kd0, kd1, key_m, split, G, relin_tol, relin_tol_state)

The factor over a slice's readings. `G` holds per reading (in the order given) its physical sensitivity ∂h/∂x, taken once at
the reading's first prediction (`slice_sensitivity`): Σ is fixed by it, as the dynamics factor's whitener is fixed at its
initial prediction.
"""
function SliceObservationFactor(model::SystemModel, readings::Vector{SliceReading}, key_z::VarKey, key_y0::VarKey, t0::Float64, first::Bool;
                                key_y1::Union{Nothing,VarKey}=nothing, t1::Float64=NaN, kd0, kd1, key_m=nothing, split=nothing,
                                G::AbstractMatrix, relin_tol::Float64=1e-3, relin_tol_state::Float64=relin_tol, chunk::Int=12)
    isempty(readings) && throw(ArgumentError("a slice factor needs at least one reading"))
    closed = key_y1 !== nothing
    closed && !(t1 > t0) && throw(ArgumentError("a closed slice ends after it starts"))
    all(r -> r.t >= t0 - 1e-9 && (!closed || r.t <= t1 + 1e-9), readings) || throw(ArgumentError("a slice's readings lie in its interval"))
    ord = sortperm([r.t for r in readings]); readings = readings[ord]; kd0 = kd0[ord]; kd1 = kd1[ord]
    m = length(readings); bl = model.bl
    Δ = closed ? t1 - t0 : NaN
    τ = [r.t - t0 for r in readings]
    a = closed ? τ ./ Δ : zeros(m)
    size(G) == (m, bl.n) || throw(DimensionMismatch("one sensitivity row per reading"))
    G = Matrix{Float64}(G)[ord, :]
    Q1 = belief_process_cov(bl, model.sn, 1.0)                                   # per hour, physical
    K = [closed ? (min(τ[i], τ[j]) - τ[i] * τ[j] / Δ) / 3600 : min(τ[i], τ[j]) / 3600 for i in 1:m, j in 1:m]
    Σ = (G * Q1 * G') .* K
    # the discrepancy: per channel the OU conditional on its values at t0 (and t1)
    αd = zeros(m); βd = zeros(m)
    for c in unique([r.spec.id for r in readings])
        ix = [i for i in 1:m if readings[i].spec.id === c]
        nm = readings[first_index(ix)].spec.noise
        has_discrepancy(nm) || continue
        ts = [readings[i].t for i in ix]
        anchors = closed ? [t0, t1] : [t0]
        Caa = [_ou_cov(nm, s, t) for s in anchors, t in anchors]
        Cra = [_ou_cov(nm, s, t) for s in ts, t in anchors]
        Crr = [_ou_cov(nm, s, t) for s in ts, t in ts]
        B = Cra / Symmetric(Caa)
        αd[ix] = B[:, 1]; closed && (βd[ix] = B[:, 2])
        Σ[ix, ix] .+= Crr - B * Cra'
    end
    for (i, r) in enumerate(readings); Σ[i, i] += r.σ^2; end
    Σ = Matrix(Symmetric(Σ))
    Lw = Matrix(inv(cholesky(Symmetric(_stable_cov(Σ))).L))
    keys = key_m === nothing ? VarKey[key_z] : VarKey[key_z, key_m]
    push!(keys, key_y0); closed && push!(keys, key_y1)
    for k in vcat(kd0, kd1)
        k === nothing || k in keys || push!(keys, k)
    end
    times = unique([r.t for r in readings if r.t > t0 + 1e-9])
    closed && (isempty(times) || last(times) < t1 - 1e-9) && push!(times, t1)
    return SliceObservationFactor(model, key_z, key_m, split, key_y0, key_y1, t0, t1, first, readings, kd0, kd1, a, αd, βd,
                                  G, Σ, Lw, keys, times, chunk, relin_tol, relin_tol_state, nothing, 0, 0)
end
"A reading's physical sensitivity ∂h/∂x at the belief state `y` (coordinates) and latents `z`."
function slice_sensitivity(model::SystemModel, r::SliceReading, y::AbstractVector, z::AbstractVector)
    bl = model.bl
    p, _, tk = _point_params(model, z, r.t)
    hx(xb) = r.h(system_from_belief(xb, bl), z, p, tk, r.t)
    return ForwardDiff.gradient(hx, belief_decode(y, bl))
end
"The same factor over a subset of its readings (their fixed rows of Σ and G): the arrival's readings set aside."
function _slice_subset(f::SliceObservationFactor, idx::AbstractVector{Int})
    isempty(idx) && return nothing
    Σ = f.Σ[idx, idx]
    Lw = Matrix(inv(cholesky(Symmetric(_stable_cov(Σ))).L))
    kd0 = f.kd0[idx]; kd1 = f.kd1[idx]
    keys = f.key_m === nothing ? VarKey[f.key_z] : VarKey[f.key_z, f.key_m]
    push!(keys, f.key_y0); f.key_y1 === nothing || push!(keys, f.key_y1)
    for k in vcat(kd0, kd1)
        k === nothing || k in keys || push!(keys, k)
    end
    rd = f.readings[idx]
    times = unique([r.t for r in rd if r.t > f.t0 + 1e-9])
    f.key_y1 !== nothing && (isempty(times) || last(times) < f.t1 - 1e-9) && push!(times, f.t1)
    return SliceObservationFactor(f.model, f.key_z, f.key_m, f.split, f.key_y0, f.key_y1, f.t0, f.t1, f.first, rd, kd0, kd1,
                                  f.a[idx], f.αd[idx], f.βd[idx], f.G[idx, :], Σ, Lw, keys, times, f.chunk, f.relin_tol, f.relin_tol_state, nothing, 0, 0)
end
first_index(ix) = ix[1]

# the latents, anchor, end node and a discrepancy lookup from the factor's values (in `vars` order)
function _slice_unpack(f::SliceObservationFactor, vals)
    d = Dict(k => v for (k, v) in zip(f.keys, vals))
    z = f.key_m === nothing ? d[f.key_z] : assemble_latents(f.split, d[f.key_z], d[f.key_m])
    y0 = d[f.key_y0]; y1 = f.key_y1 === nothing ? nothing : d[f.key_y1]
    return z, y0, y1, d
end

# the slice's solve at (z, y0), cached by point; the Jacobian of every saved state when asked, kept within the tolerances
function _slice_eval!(f::SliceObservationFactor, z, y0; jac::Bool)
    c = f.cache
    ny = length(y0)
    if c === nothing || c.z != z || c.y0 != y0
        tq = time(); S = propagate_states(f.model, y0, z, f.t0, f.times; first_slice=f.first); _tick!(:slice_solve, time() - tq)
        f.n_solves += 1
        c = (z=copy(z), y0=copy(y0), S=S, jz=c === nothing ? nothing : c.jz, jy0=c === nothing ? nothing : c.jy0,
             Jy=c === nothing ? nothing : c.Jy, Jz=c === nothing ? nothing : c.Jz)
    end
    if jac && !isempty(f.times) && (c.Jy === nothing || norm(z .- c.jz, Inf) > f.relin_tol || norm(y0 .- c.jy0, Inf) > f.relin_tol_state)
        g(v) = vec(propagate_states(f.model, v[1:ny], v[(ny + 1):end], f.t0, f.times; first_slice=f.first))
        tq = time()
        J = _chunked_jacobian(g, vcat(y0, z), ny * length(f.times), f.chunk)
        _tick!(:slice_jac, time() - tq)
        f.n_jacobians += 1
        c = (z=c.z, y0=c.y0, S=c.S, jz=copy(z), jy0=copy(y0), Jy=J[:, 1:ny], Jz=J[:, (ny + 1):end])
    end
    f.cache = c
    return c
end
_slice_col(f::SliceObservationFactor, t) = t <= f.t0 + 1e-9 ? 0 : findfirst(τ -> abs(τ - t) < 1e-9, f.times)

"A slice reading uses the same latent-dependent ROM correction as a node observation."
function _slice_observation(model::SystemModel, r::SliceReading, y, z)
    _validate_rom_use(model.rom, model.blk, z, r.t, r.t)
    p, _, tk = _point_params(model, z, r.t)
    pred = r.h(system_from_belief(belief_decode(y, model.bl), model.bl), z, p, tk, r.t)
    key = model.rom === nothing ? nothing : _rom_key(r.spec.observable)
    bias = key === nothing ? r.bias : _rom_observation_bias(model.rom, r.spec.observable, key, r.t, z, model.blk)
    return pred - bias
end

"The readings' predictions μ and, with `jac`, their Jacobian per variable (a matrix per key in `vars` order)."
function _slice_predict(f::SliceObservationFactor, vals; jac::Bool=false)
    z, y0, y1, d = _slice_unpack(f, vals)
    c = _slice_eval!(f, z, y0; jac)
    ny = length(y0); nz = length(z); bl = f.model.bl
    m = length(f.readings)
    closed = y1 !== nothing
    iend = closed ? length(f.times) : 0
    Φend = closed ? c.S[:, iend] : nothing
    μ = zeros(promote_type(map(eltype, vals)..., eltype(c.S)), m)
    Js = jac ? Dict(k => zeros(m, length(v)) for (k, v) in zip(f.keys, vals)) : nothing
    for (i, r) in enumerate(f.readings)
        j = _slice_col(f, r.t)
        Φ = j == 0 ? y0 : c.S[:, j]
        y = closed ? Φ .+ f.a[i] .* (y1 .- Φend) : Φ
        μ[i] = _slice_observation(f.model, r, y, z)
        f.kd0[i] === nothing || (μ[i] += f.αd[i] * d[f.kd0[i]][1])
        f.kd1[i] === nothing || (μ[i] += f.βd[i] * d[f.kd1[i]][1])
        jac || continue
        # ∂h at (y, z): the direct dependence on the latents (p) and on the state
        hw = function (w)
            zz = w[1:nz]; yy = w[(nz + 1):end]
            return _slice_observation(f.model, r, yy, zz)
        end
        gw = ForwardDiff.gradient(hw, vcat(z, y))
        gz = gw[1:nz]; gy = gw[(nz + 1):end]
        rows(J, jj) = J[((jj - 1) * ny + 1):(jj * ny), :]
        Jy0 = j == 0 ? Matrix{Float64}(I, ny, ny) : rows(c.Jy, j)
        Jz0 = j == 0 ? zeros(ny, nz) : rows(c.Jz, j)
        if closed
            Jy0 = Jy0 .- f.a[i] .* rows(c.Jy, iend); Jz0 = Jz0 .- f.a[i] .* rows(c.Jz, iend)
            Js[f.key_y1][i, :] .+= f.a[i] .* gy
        end
        Js[f.key_y0][i, :] .+= vec(gy' * Jy0)
        dz = vec(gy' * Jz0) .+ gz
        if f.key_m === nothing
            Js[f.key_z][i, :] .+= dz
        else
            Js[f.key_z][i, :] .+= dz[f.split.static]; Js[f.key_m][i, :] .+= dz[f.split.moving]
        end
        f.kd0[i] === nothing || (Js[f.kd0[i]][i, 1] += f.αd[i])
        f.kd1[i] === nothing || (Js[f.kd1[i]][i, 1] += f.βd[i])
    end
    return μ, jac ? [Js[k] for k in f.keys] : nothing
end
_slice_values(f::SliceObservationFactor) = [r.value for r in f.readings]
function _ensure_current!(f::SliceObservationFactor, g::FactorGraph)
    c = f.cache
    c === nothing && return nothing
    z, y0, _, _ = _slice_unpack(f, [g.values[k] for k in f.keys])
    if c.Jy === nothing || c.jz != z || c.jy0 != y0
        f.cache = (z=c.z, y0=c.y0, S=c.S, jz=nothing, jy0=nothing, Jy=nothing, Jz=nothing)
    end
    return nothing
end
residual(f::SliceObservationFactor, vals) = f.Lw * (_slice_values(f) .- first(_slice_predict(f, vals)))
function jacobians(f::SliceObservationFactor, vals)
    _, Jμ = _slice_predict(f, vals; jac=true)
    return [-f.Lw * J for J in Jμ]
end
function factor_linearization(f::SliceObservationFactor, vals)
    μ, Jμ = _slice_predict(f, vals; jac=true)
    return (r=f.Lw * (_slice_values(f) .- μ), Js=[-f.Lw * J for J in Jμ])
end
loss(f::SliceObservationFactor, vals) = 0.5 * sum(abs2, residual(f, vals))
"The raw innovations (reading less prediction) at the current values."
slice_innovations(f::SliceObservationFactor, vals) = _slice_values(f) .- first(_slice_predict(f, vals))
export SliceObservationFactor, slice_innovations

# A dynamics factor over the same interval as a closed slice reads the slice's one solve and Jacobian: its end state is the
# slice solve's last column (saved at the next node), its Jacobian that column's rows. Its whitener stays its own (fixed at
# its first prediction); the slice's outputs for the past's integrals are solved lazily (`last_solve`).
function _dyn_eval_peer!(f::DynamicsFactor, p::SliceObservationFactor, z, y0; jac::Bool)
    (p.key_y1 == f.key_y1 && abs(p.t1 - f.t1) < 1e-9 && abs(last(p.times) - f.t1) < 1e-9) ||
        throw(ArgumentError("a dynamics factor's peer must be the closed slice over its own interval"))
    cs = _slice_eval!(p, z, y0; jac)
    ny = length(y0); j = length(p.times)
    fy = cs.S[:, j]
    c = f.cache
    W = c === nothing ? _whitener(f, fy) : c.W
    wy = c === nothing ? copy(fy) : c.wy
    o = (c !== nothing && c.z == z && c.y0 == y0) ? c.o : nothing
    Jy = cs.Jy === nothing ? nothing : cs.Jy[((j - 1) * ny + 1):(j * ny), :]
    Jz = cs.Jz === nothing ? nothing : cs.Jz[((j - 1) * ny + 1):(j * ny), :]
    f.cache = (z=copy(z), y0=copy(y0), f=fy, W=W, wy=wy, o=o, jz=cs.jz, jy0=cs.jy0, Jy=Jy, Jz=Jz)
    return f.cache
end
