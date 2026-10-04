using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, LinearAlgebra, Test
const E = BrisketEstimator

@testset "joint accuracy includes narrow correlated directions" begin
    A = [1e6 -1e6; 1.0 1.0]
    key = VarKey(:state, :joint_accuracy, 1)
    function joint_graph(target)
        graph = FactorGraph()
        add_variable!(graph, key, zeros(2))
        add_factor!(graph, LinearFactor([key], [A], A * target))
        graph
    end
    # Marginal widths are about 0.5, but this direction is tightly constrained.
    # A 1e-5 coordinate error is twenty JOINT SDs, despite tiny marginal errors.
    narrow = [1e-5, -1e-5]
    g = joint_graph(narrow)
    r = gauss_newton!(g; iters=1, tol=1e-12, joint_tol=1e-3, max_step=1e-9, soc=0)
    @test !r.converged
    @test r.newton_sd < 1e-3
    @test r.newton_joint_sd > 19.0
    @test E._stationarity_metrics(A, -A * narrow, ones(2))[2] ≈ 20.0

    # The same order of coordinate error in the broad direction is harmless.
    # The caller explicitly requests accuracy relative to the joint posterior.
    broad = [1e-4, 1e-4]
    g = joint_graph(broad)
    r = gauss_newton!(g; tol=1e-12, joint_tol=1e-3)
    @test r.converged && r.gradient > 1e-12
    @test r.newton_joint_sd ≈ 2e-4
    @test r.hist[end].joint_converged
    @test variable_value(g, key) == zeros(2)
    @test maximum(abs.(broad) ./ sqrt.(diag(joint_cov(g)[1]))) <= r.joint_tol

    # Generic callers retain the original stopping rule unless they opt in.
    disabled = gauss_newton!(joint_graph(broad); iters=1, tol=1e-12, max_step=1e-9, soc=0)
    @test !disabled.converged
    sparse_result = gauss_newton!(joint_graph(broad); tol=1e-12, joint_tol=1e-3, sparse=true)
    @test sparse_result.converged && sparse_result.newton_joint_sd ≈ 2e-4
    for invalid in (-1.0, Inf, NaN)
        @test_throws ArgumentError gauss_newton!(g; joint_tol=invalid)
    end
end
