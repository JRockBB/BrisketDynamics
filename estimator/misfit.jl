# The misfit check: what a refit reports next to its posterior. Per
# channel, the whitened residual power against its expectation under the
# noise model, and the residuals' lag-1 autocorrelation — the two
# statistics a structurally wrong model cannot hide from (the 2-D
# cylinder plant of docs/results.md was 17 sd wrong with a clean-looking
# posterior; its objective per reading was the only tell). Included by
# BrisketEstimator after channels.jl.
#
# With the model and the noise right, the whitened residuals r = w − a(ẑ)
# are white with unit variance up to the fit's own absorption: the hat
# matrix H = B V Bᵀ (V the posterior covariance, prior included) takes
# tr(H_c) effective degrees of freedom from channel c, so
# E[Σ_c r²] ≈ n_c − tr(H_c) and Var ≈ 2 (n_c − tr(H_c)). The lag-1
# autocorrelation of n white residuals has sd ≈ 1/√n. Both are quoted as
# z-scores; both are approximate (a fixed truth under a proper prior
# leaves a small shrinkage bias, and the lag-1 statistic ignores the hat
# matrix), which is the right precision for a warning light.

export misfit, channel_label

channel_label(::PitProbe) = "pit"
channel_label(::MeatProbe) = "meat probe"
channel_label(::SurfaceIR) = "surface IR"
channel_label(::HumiditySensor) = "humidity"
channel_label(::MassScale) = "scale"
channel_label(ch::StaticReading) = "static " * string(ch.latent)

# the per-block statistics on a residual vector and its hat-matrix diagonal
function _block_stats(r::AbstractVector, h::AbstractVector)
    n = length(r)
    ss = sum(abs2, r)
    dof = n - sum(h)
    z_chi2 = dof > 0 ? (ss - dof) / sqrt(2 * dof) : NaN
    rho1 = n > 1 && ss > 0 ? sum(r[i] * r[i + 1] for i in 1:(n - 1)) / ss : NaN
    z_rho1 = n > 1 ? rho1 * sqrt(n) : NaN
    return (n=n, ss=ss, dof=dof, z_chi2=z_chi2, rho1=rho1, z_rho1=z_rho1)
end

"""
    misfit(channels, a, B, w, V) -> (channels=[...], total=...)

The misfit check on the assembled (whitened) residuals `w − a` with the
affine rows `B` and the posterior covariance `V`: one record per channel
in registry order (channels with no rows are skipped, as `assemble`
skips them) and one for every row together. Each record carries the row
count, the residual power, its effective degrees of freedom, the χ²
z-score and the lag-1 autocorrelation with its z-score.
"""
function misfit(channels, a, B, w, V)
    r = w .- a
    h = [dot(view(B, i, :), V, view(B, i, :)) for i in axes(B, 1)]   # diag(B V Bᵀ)
    out = NamedTuple[]
    row = 0
    for ch in channels
        nr = size(whitening(ch), 1)
        nr == 0 && continue
        idx = (row + 1):(row + nr)
        push!(out, (name=channel_label(ch), _block_stats(view(r, idx), view(h, idx))...))
        row += nr
    end
    row == length(r) || throw(ArgumentError("channel rows ($row) do not cover the residual ($(length(r)))"))
    return (channels=out, total=(name="total", _block_stats(r, h)...))
end
