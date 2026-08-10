# Piecewise-linear interpolation on a *uniform* knot grid, implemented as a
# branch-free hat-basis sum. The operation sequence is independent of `t`
# (no data-dependent index lookup), which makes it safe inside a ReverseDiff
# *compiled* tape — a `searchsorted`-style implementation would freeze the
# branch taken at tape-compile time and produce silently wrong adjoints.

"""
    lininterp(t, t0, dt, vals)

Linear interpolation of knot values `vals` on the uniform grid
`t0 + (i-1)*dt`, evaluated at time `t`. Constant extrapolation outside the
grid. O(length(vals)) per call, branch-free.
"""
function lininterp(t, t0, dt, vals::AbstractVector)
    n = length(vals)
    τ = clamp(t, t0, t0 + (n - 1) * dt)
    s = zero(τ * first(vals))
    @inbounds for i in 1:n
        w = clamp(1 - abs(τ - (t0 + (i - 1) * dt)) / dt, 0, 1)
        s += w * vals[i]
    end
    s
end

"""
    hat_matrix(ts, t0, dt, n)

Dense interpolation matrix `M` (length(ts) × n) with plain `Float64` entries
such that `M * vals` interpolates the knots at all times `ts` at once. Used on
the output/observable path where a matrix–vector product is far friendlier to
Zygote than a scalar loop.
"""
function hat_matrix(ts::AbstractVector, t0, dt, n::Int)
    M = zeros(length(ts), n)
    for (j, t) in enumerate(ts)
        τ = clamp(t, t0, t0 + (n - 1) * dt)
        for i in 1:n
            M[j, i] = clamp(1 - abs(τ - (t0 + (i - 1) * dt)) / dt, 0, 1)
        end
    end
    M
end
