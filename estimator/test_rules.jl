# Regression tests for the RxInfer delta-node behaviours the estimator's
# graph structure depends on (found 2026-09-07 on RxInfer 5.5.2; see
# docs/results.md, "RxInfer estimator"). Linear cases only, so every
# expected posterior is exact and the run is seconds after compile.
#
#   1. a deterministic node with more outputs than latent inputs SILENTLY
#      returns the prior (singular forward output covariance) — which is
#      why the graph uses one node per residual;
#   2. many per-residual nodes sharing TWO latent blocks come back with the
#      right means and variances that are NOT the true marginals (a block
#      factorization across the nodes, not the joint; ~1 % off in this
#      weakly correlated design, 2-5x too narrow on the real problem's
#      ridge) — which is why the latents are one multivariate block;
#   3. per-residual affine nodes on ONE block, with initialized messages
#      and a few iterations, are exact.
#
# If a future RxInfer release changes 1 or 2, these tests say so and the
# workarounds can be reconsidered.
#
# Run:  julia --project=estimator estimator/test_rules.jl
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
using RxInfer, LinearAlgebra, Random, Test

# --- 1. one vector input, three outputs -------------------------------------
const A3 = [1.0 0.0; 1.0 1.0; 0.0 2.0]
const mz = [0.0, 0.0]
const Vz = Matrix(4.0 * I, 2, 2)
const R3 = Matrix(0.25 * I, 3, 3)
const obs3 = [1.0, 3.0, 4.0]
const P3 = inv(inv(Vz) + A3' * inv(R3) * A3)
const m3 = P3 * (inv(Vz) * mz + A3' * inv(R3) * obs3)

flin(z) = A3 * z
@model function one_node(obs, mz, Vz, R)
    z ~ MvNormalMeanCovariance(mz, Vz)
    y := flin(z)
    obs ~ MvNormalMeanCovariance(y, R)
end
meta1 = @meta begin
    flin() -> Linearization()
end

# --- 2 & 3. many scalar readings of a 2-dim latent -------------------------
const NREAD = 37
Random.seed!(7)
const AW = randn(NREAD, 2)
const yw = AW * [0.7, -1.2] .+ 0.5 .* randn(NREAD)
const s2 = 0.25
const PW = inv(inv(Vz) + AW' * AW ./ s2)
const mW = PW * (inv(Vz) * mz + AW' * yw ./ s2)

row1(z, i) = dot(AW[i, :], z)
@model function readings_one_block(obs, mz, Vz, s2)
    z ~ MvNormalMeanCovariance(mz, Vz)
    for i in eachindex(obs)
        pred[i] := row1(z, i)
        obs[i] ~ Normal(mean = pred[i], var = s2)
    end
end
meta_one = @meta begin
    row1() -> Linearization()
end
@initialization function init_one(mz, Vz)
    μ(z) = MvNormalMeanCovariance(mz, Vz)
end

row2(za, zb, i) = AW[i, 1] * za + AW[i, 2] * zb
@model function readings_two_blocks(obs, ma, va, mb, vb, s2)
    za ~ Normal(mean = ma, var = va)
    zb ~ Normal(mean = mb, var = vb)
    for i in eachindex(obs)
        pred[i] := row2(za, zb, i)
        obs[i] ~ Normal(mean = pred[i], var = s2)
    end
end
meta_two = @meta begin
    row2() -> Linearization()
end
@initialization function init_two(ma, va, mb, vb)
    μ(za) = NormalMeanVariance(ma, va)
    μ(zb) = NormalMeanVariance(mb, vb)
end

@testset "RxInfer delta-node rules the estimator relies on" begin
    # 1. outputs (3) > latent inputs (2): the prior comes back untouched
    r = infer(model = one_node(mz = mz, Vz = Vz, R = R3), data = (obs = obs3,), meta = meta1)
    m, V = mean_cov(r.posteriors[:z])
    @test m ≈ mz atol = 1e-12
    @test Matrix(V) ≈ Vz atol = 1e-12
    @test !isapprox(m, m3; atol = 1e-3)

    # 3. per-reading nodes, one block: exact
    r = infer(model = readings_one_block(mz = mz, Vz = Vz, s2 = s2), data = (obs = yw,),
              meta = meta_one, initialization = init_one(mz, Vz), iterations = 20,
              returnvars = (z = KeepLast(),))
    m, V = mean_cov(r.posteriors[:z])
    @test m ≈ mW atol = 1e-8
    @test Matrix(V) ≈ PW atol = 1e-8

    # 2. per-reading nodes, two blocks: means right, variances not the marginals
    r = infer(model = readings_two_blocks(ma = 0.0, va = 4.0, mb = 0.0, vb = 4.0, s2 = s2),
              data = (obs = yw,), meta = meta_two, initialization = init_two(0.0, 4.0, 0.0, 4.0),
              iterations = 20, returnvars = (za = KeepLast(), zb = KeepLast()))
    ma_, va_ = mean(r.posteriors[:za]), var(r.posteriors[:za])
    mb_, vb_ = mean(r.posteriors[:zb]), var(r.posteriors[:zb])
    println("two blocks, $NREAD readings: sd ratio vs exact = ",
            round.([sqrt(va_ / PW[1, 1]), sqrt(vb_ / PW[2, 2])]; digits=3))
    @test [ma_, mb_] ≈ mW atol = 1e-6
    @test !(isapprox(va_, PW[1, 1]; rtol = 1e-3) && isapprox(vb_, PW[2, 2]; rtol = 1e-3))
end
