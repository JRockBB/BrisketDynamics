# Piecewise-linear interpolation on a *uniform* knot grid, implemented as a
# branch-free hat-basis sum. The operation sequence is independent of `t`
# (no data-dependent index lookup), which makes it safe under any AD backend
# *compiled* tape — a `searchsorted`-style implementation would freeze the
# branch taken at tape-compile time and produce silently wrong adjoints.

"""
    lininterp(t::Number, t0::Number, dt::Number, vals::AbstractVector)

Linear interpolation of knot values `vals` on the uniform grid
`t0 + (i-1)·dt`, evaluated at time `t`. Constant extrapolation outside the
grid. O(length(vals)) per call and branch-free: the full hat-basis sum runs
every call so the operation sequence never depends on `t`.

# Arguments
- `t::Number`: Evaluation time [s].
- `t0::Number`: First knot time [s].
- `dt::Number`: Uniform knot spacing [s].
- `vals::AbstractVector`: Knot values.

# Returns
- `s::Number`: Interpolated value at `t`.
"""
function lininterp(t, t0, dt, vals::AbstractVector)
    n = length(vals)
    τ = clamp(t, t0, t0 + (n - 1) * dt)
    s = zero(τ * first(vals))
    @inbounds for i in 1:n
        w = clamp(1 - abs(τ - (t0 + (i - 1) * dt)) / dt, 0, 1)
        s += w * vals[i]
    end
    return s
end

"""
    hat_matrix(ts::AbstractVector, t0::Number, dt::Number, n::Int)

Dense hat-basis interpolation matrix with plain `Float64` entries such that
`M * vals` interpolates the knots at all times `ts` at once. Used on the
output/observable path where a matrix–vector product is far friendlier to
forward-mode AD than a scalar loop.

# Arguments
- `ts::AbstractVector`: Evaluation times [s].
- `t0::Number`: First knot time [s].
- `dt::Number`: Uniform knot spacing [s].
- `n::Int`: Number of knots.

# Returns
- `M::Matrix{Float64}`: Interpolation matrix, size `(length(ts), n)`.
"""
function hat_matrix(ts::AbstractVector, t0, dt, n::Int)
    M = zeros(length(ts), n)
    for (j, t) in enumerate(ts)
        τ = clamp(t, t0, t0 + (n - 1) * dt)
        for i in 1:n
            M[j, i] = clamp(1 - abs(τ - (t0 + (i - 1) * dt)) / dt, 0, 1)
        end
    end
    return M
end
