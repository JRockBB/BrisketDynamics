# The physics-selectable stack against the mock plants, where the answer
# is known: at the true latents the 1-D stack reproduces the monolith
# mock's clean readings through the registry's channels (pit, core at
# the true depth, IR, humidity, mass), the 2-D stack reproduces the
# cylinder mock's (same engine, same grid), the radius column of the 2-D
# Jacobian matches a directional ForwardDiff derivative, and the mass
# block's area column matches the old explicit factor.
#
# Run:  julia --project=estimator estimator/test/test_physics2d.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using ForwardDiff, JSON, LinearAlgebra, Test

const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, "..", ".."))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
prior_phys = read_sites(gate["prior"])
names = String.(gate["names"])

function setup(path)
    m = JSON.parsefile(path)
    t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
    pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
    cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
    d = estimator_setup(t, u, pit, cfg, prior_phys, names)
    x = Float64.(m["truth"]["x"])            # 15 physics, area, depth fraction
    return m, d, x
end

# z for the physics from x (the truth), plus the estimator-only latents appended
zphys(x, prior) = [to_z(v, p) for (v, p) in zip(x[1:15], prior[1:15])]

@testset "1-D stack reproduces the monolith mock's clean readings" begin
    m, d, x = setup(joinpath(REPO, "data", "mock", "mock_1d.json"))
    A_true, depth_fraction_true = x[16], x[17]
    depth_site = probe_depth_site()
    z = vcat(zphys(x, prior_phys), log(A_true), to_z(depth_fraction_true, depth_site))
    prior = vcat(prior_phys, [PriorSite("lognormal", log(A_true), 0.3), depth_site])
    S, J, lay = physics_outputs(z, d, Physics1D(area_latent=16))
    @test length(S) == length(lay) == size(J, 1) && size(J, 2) == 17
    full = trues(length(d.t))
    clean = m["clean"]
    for (ch, key) in ((PitProbe(1.0, 0.0, full), "pit"), (MeatProbe(1.0, 0.0, full; depth_latent=17), "core"),
                      (SurfaceIR(1.0, 0.0, full), "ir"), (HumiditySensor(1.0, 0.0, full), "humidity"),
                      (MassScale(1.0, 0.0, full), "mass"))
        r, _ = E.predict(ch, S, J, z, prior, lay)
        rel = maximum(abs.(r .- Float64.(clean[key]))) / maximum(abs.(Float64.(clean[key])))
        println("  1-D ", key, ": max rel diff ", rel)
        @test rel < 1e-4          # the adaptive solve's tolerance (1e-6) shows as ~1e-5 in the outputs
    end
    # the mass block's area column: d mass / d log A at fixed everything else is the mass itself
    idx = (lay.oMass + 1):(lay.oMass + lay.nt)
    # the explicit factor plus the pit-coupling part; the explicit part alone equals S (mass ∝ A)
    dm = ForwardDiff.derivative(a -> physics_stack(vcat(z[1:15], a, z[17:end]), d, Physics1D(area_latent=16))[1][idx][end],
                                log(A_true))
    @test isapprox(J[idx[end], 16], dm; rtol=2e-3)
end

@testset "2-D stack reproduces the cylinder mock's clean readings" begin
    m, d, x = setup(joinpath(REPO, "data", "mock", "mock_2d.json"))
    g = m["geometry"]
    R_true, depth_fraction_true = Float64(g["R"]), x[17]
    ph = Physics2D(radius_latent=16, nr=Int(g["nr"]), nz=Int(g["nz"]), h=30.0, n_meat=1.0, nx_line=Int(g["nz"]))
    depth_site = probe_depth_site()
    z = vcat(zphys(x, prior_phys), log(R_true), to_z(depth_fraction_true, depth_site))
    prior = vcat(prior_phys, [PriorSite("lognormal", log(R_true), 0.3), depth_site])
    S, lay = physics_stack(z, d, ph)
    @test length(S) == length(lay)
    full = trues(length(d.t))
    clean = m["clean"]
    J0 = zeros(length(S), 17)
    for (ch, key, tol) in ((PitProbe(1.0, 0.0, full), "pit", 1e-6), (SurfaceIR(1.0, 0.0, full), "ir", 1e-6),
                           (HumiditySensor(1.0, 0.0, full), "humidity", 1e-6), (MassScale(1.0, 0.0, full), "mass", 1e-6),
                           (MeatProbe(1.0, 0.0, full; depth_latent=17), "core", 2e-3))
        r, _ = E.predict(ch, S, J0, z, prior, lay)
        rel = maximum(abs.(r .- Float64.(clean[key]))) / maximum(abs.(Float64.(clean[key])))
        println("  2-D ", key, ": max rel diff ", rel)
        @test rel < tol
    end
    # the Jacobian: the radius column against a directional derivative through the stack
    S2, J, lay2 = physics_outputs(z, d, ph)
    @test S2 == S
    dR = ForwardDiff.derivative(lr -> physics_stack(vcat(z[1:15], lr, z[17:end]), d, ph)[1], log(R_true))
    @test isapprox(J[:, 16], dR; rtol=1e-6, atol=1e-8 * maximum(abs, dR))
    @test !all(iszero, J[:, 8])         # m_conv scales the 2-D engine's h_c and k_m with the draft, as in 1-D
    @test !all(iszero, J[:, 4])         # L is (the half-height)
end
println("test_physics2d.jl: done")
