# The innovation autocovariances' sensitivity to the state's process-noise rates, accumulated along a filter
# run, for the bench's identification of those rates from the innovations alone: the autocovariance
# least-squares method (Odelson, Rajamani, Rawlings 2006) in its fixed-gain, time-varying form (Åkesson,
# Jørgensen, Poulsen, Jørgensen 2008; Zhang, Sjövall et al. 2020, section IV).
#
# The filter's update at an arrival is, to first order, a Kalman update with the rows H (the affine model at the
# predicted mean, one row per reading, whitened by the reading's noise) and the gain K = P⁻ Hᵀ S⁻¹ on the
# predicted innovation covariance S = H P⁻ Hᵀ + I; its prediction between arrivals is a transition F (the sigma
# points' regression, or the tangent) with the process noise Q_known + Σ_g θ_g Q̇_g, θ_g the square of group g's
# rate and Q̇_g the group's covariance per unit of it in the state's coordinate at the interval's end. Under its
# own model the filter's innovations are white with covariance S at every update. Under the rates θ + δθ, with
# the gains the filter actually used, the prediction error carries an extra covariance Π_g per unit δθ_g,
#   Π_g ← F Π_g Fᵀ + Q̇_g  at a prediction,   Π_g ← (I − K H) Π_g (I − K H)ᵀ  at an update,
# and the innovation autocovariance between the update at t and an earlier one at t′ is
#   E[e(t) e(t′)ᵀ] = [t = t′] S + Σ_g δθ_g H(t) Φ(t, t′) Π_g(t′) H(t′)ᵀ,
# Φ the product of the closed-loop transitions F (I − K H) between the two. In units standardized by the filter's
# predicted innovation sds this is a linear regression of the innovations' products on the δθ_g, and the
# accumulator carries its sufficient statistics: the normal equations, per channel pair and kind (the squares at
# one update, the products across updates) so the driver can weight each by its own residual variance (the
# method's weighting, feasible from the rows themselves: a probe whose innovations are tiny against its
# predicted spread has products a hundredth as scattered as another's), and per channel pair and time lag the
# sample products, their squares, the base term and each group's coefficient, for the driver to pool over the
# cooks, solve for δθ ≥ −θ, and show. The initial widths are not in it: their transient is one, the rates recur.

"The normal equations of one block of rows: the products of one channel pair of one kind (the same update, or across updates)."
mutable struct ALSBlock
    G::Matrix{Float64}
    h::Vector{Float64}
    rss::Float64
    n::Int
    s_sum::Float64
    s_sq::Float64
end
ALSBlock(n_g::Int) = ALSBlock(zeros(n_g, n_g), zeros(n_g), 0.0, 0, 0.0, 0.0)

"""
    ALSAccumulator(groups, theta, n, n_ch; lag_s=7200.0, bin_s=120.0, keep=false)

The accumulator of one run: the groups' names and the variance rates θ (the squares of the rates the run uses),
the block's width `n` and the channel count; pairs of updates farther apart than `lag_s` seconds are not scored;
the pairs' statistics are binned by their time lag in `bin_s`; `keep` stores every update's linear pieces (the
tests replay them).
"""
mutable struct ALSAccumulator
    groups::Vector{String}
    theta::Vector{Float64}
    n::Int
    n_ch::Int
    lag_s::Float64
    bin_s::Float64
    Pi::Vector{Matrix{Float64}}                    # per group: the prediction error's covariance per unit of its variance rate
    pending::Vector{Any}                           # the updates inside the lag window: t, the rows' channels, the standardized innovations (a column per replicate), S, and per group the propagated Π H′ᵀ
    G::Matrix{Float64}                             # the normal equations of the regression, every row at unit weight
    h::Vector{Float64}
    rss::Float64                                   # the products' sum of squares less the base: the regression's cost at δθ = 0
    n_rows::Int
    n_updates::Int
    pairs::Dict{Tuple{Int,Int,Int},Vector{Float64}}   # (channel at t, channel at t′, lag bin) → [count, Σ product, Σ product², Σ base, Σ coefficient per group]
    blocks::Dict{Tuple{Int,Int,Bool},ALSBlock}     # (channel at t, channel at t′, the same update) → that block's normal equations
    keep::Bool
    pieces::Vector{Any}
end
function ALSAccumulator(groups::AbstractVector{<:AbstractString}, theta::AbstractVector, n::Int, n_ch::Int;
                        lag_s::Float64=7200.0, bin_s::Float64=120.0, keep::Bool=false)
    n_g = length(groups)
    length(theta) == n_g || throw(ArgumentError("one variance rate per group ($(length(theta)) for $n_g groups)"))
    all(θ -> θ >= 0, theta) || throw(ArgumentError("a variance rate is negative"))
    lag_s > 0 && bin_s > 0 || throw(ArgumentError("the lag window and the bin must be positive"))
    return ALSAccumulator(collect(String, groups), collect(Float64, theta), n, n_ch, lag_s, bin_s,
                          [zeros(n, n) for _ in 1:n_g], Any[], zeros(n_g, n_g), zeros(n_g), 0.0, 0, 0,
                          Dict{Tuple{Int,Int,Int},Vector{Float64}}(), Dict{Tuple{Int,Int,Bool},ALSBlock}(), keep, Any[])
end

"""
    als_update!(acc, F, dQ, H, S, K, e, rowch, t)

One update of the filter fed to the accumulator: the transition `F` (n × n) from the last posterior to this
prediction, per group the derivative `dQ[g]` (n × n) of the interval's process covariance with respect to the
group's variance rate, and at this arrival the standardized rows `H` (p × n; p may be zero), their innovation
covariance `S` (p × p, unit diagonal), the gain `K` (n × p) the filter applied on the standardized innovations
(zero when it kept the predicted belief), the standardized innovations `e` (a vector, or a p × R matrix whose
columns are independent replicates of the run under the same gains: the tests' simulation), the channel of each
row and the time.
"""
function als_update!(acc::ALSAccumulator, F::AbstractMatrix, dQ::AbstractVector, H::AbstractMatrix, S::AbstractMatrix,
                     K::AbstractMatrix, e::AbstractVecOrMat, rowch::AbstractVector{Int}, t::Real)
    n, n_g = acc.n, length(acc.groups)
    E = e isa AbstractVector ? reshape(collect(Float64, e), :, 1) : Matrix{Float64}(e)
    p, R = size(E)
    size(F) == (n, n) || throw(ArgumentError("the transition is $(size(F)), the block is $n wide"))
    length(dQ) == n_g || throw(ArgumentError("one process-covariance derivative per group"))
    size(H) == (p, n) && size(S) == (p, p) && size(K) == (n, p) && length(rowch) == p ||
        throw(ArgumentError("the rows, their covariance, the gain, the innovations and the channels disagree in size"))
    # the prediction: the sensitivities and the pending rows carried through the interval
    for g in 1:n_g
        acc.Pi[g] = Matrix(Symmetric(F * acc.Pi[g] * F' + dQ[g]))
    end
    filter!(pd -> t - pd.t <= acc.lag_s + 1e-9, acc.pending)
    for pd in acc.pending, g in 1:n_g
        pd.M[g] = F * pd.M[g]
    end
    if p > 0
        new = (t=Float64(t), rowch=collect(Int, rowch), E=E, S=Matrix{Float64}(S), M=[acc.Pi[g] * H' for g in 1:n_g])
        push!(acc.pending, new)
        b = zeros(n_g)
        for pd in acc.pending
            same = pd === new
            C = [H * pd.M[g] for g in 1:n_g]                              # p × p′ per group
            bin = round(Int, (t - pd.t) / acc.bin_s)
            for i in 1:p, i2 in 1:size(pd.E, 1)
                same && i2 < i && continue                                # one row per unordered pair at the same update
                for g in 1:n_g; b[g] = C[g][i, i2]; end
                base = same ? new.S[i, i2] : 0.0
                prods = E[i, :] .* pd.E[i2, :]                            # one product per replicate
                s_sum = sum(prods) - R * base
                s_sq = sum(abs2, prods .- base)
                bb = b * b'
                acc.G .+= R .* bb; acc.h .+= s_sum .* b; acc.rss += s_sq; acc.n_rows += R
                blk = get!(acc.blocks, (rowch[i], pd.rowch[i2], same)) do; ALSBlock(n_g); end
                blk.G .+= R .* bb; blk.h .+= s_sum .* b; blk.rss += s_sq; blk.n += R; blk.s_sum += s_sum; blk.s_sq += s_sq
                v = get!(acc.pairs, (rowch[i], pd.rowch[i2], bin)) do; zeros(4 + n_g); end
                v[1] += R; v[2] += sum(prods); v[3] += sum(abs2, prods); v[4] += R * base; v[5:end] .+= R .* b
            end
        end
        # the update: the gain the filter applied closes the loop on the sensitivities and the pending rows
        IKH = I - K * H
        for g in 1:n_g
            acc.Pi[g] = Matrix(Symmetric(IKH * acc.Pi[g] * IKH'))
        end
        for pd in acc.pending, g in 1:n_g
            pd.M[g] = IKH * pd.M[g]
        end
    end
    acc.n_updates += 1
    acc.keep && push!(acc.pieces, (t=Float64(t), F=Matrix{Float64}(F), dQ=[Matrix{Float64}(q) for q in dQ], H=Matrix{Float64}(H),
                                   S=Matrix{Float64}(S), K=Matrix{Float64}(K), E=E, rowch=collect(Int, rowch)))
    return acc
end

"""
    als_summary(acc) -> Dict

The sufficient statistics of the run for the bench driver: the groups and their variance rates, the window and
the bin, the counts, the normal equations `G` (rows) and `h` at unit weight and the cost at δθ = 0; per
(channel at t, channel at t′, the same update) that block's own normal equations, cost, count and the sum and
sum of squares of its rows; and per (channel at t, channel at t′, lag bin) the count, the sum of the
innovations' products and of their squares, the sum of the base term and the sum of each group's coefficient.
"""
function als_summary(acc::ALSAccumulator)
    keys_sorted = sort(collect(keys(acc.pairs)))
    bkeys = sort(collect(keys(acc.blocks)))
    return Dict("groups" => acc.groups, "theta" => acc.theta, "lag_s" => acc.lag_s, "bin_s" => acc.bin_s,
                "n_updates" => acc.n_updates, "n_rows" => acc.n_rows, "rss" => acc.rss,
                "G" => [acc.G[i, :] for i in 1:size(acc.G, 1)], "h" => acc.h,
                "blocks" => [(b = acc.blocks[k]; Dict("a" => k[1], "b" => k[2], "same" => k[3], "G" => [b.G[i, :] for i in 1:size(b.G, 1)], "h" => b.h,
                                                     "rss" => b.rss, "n" => b.n, "s_sum" => b.s_sum, "s_sq" => b.s_sq)) for k in bkeys],
                "pairs" => [Dict("a" => k[1], "b" => k[2], "bin" => k[3], "n" => acc.pairs[k][1], "sum_product" => acc.pairs[k][2],
                                 "sum_product2" => acc.pairs[k][3], "sum_base" => acc.pairs[k][4], "sum_b" => acc.pairs[k][5:end]) for k in keys_sorted])
end
