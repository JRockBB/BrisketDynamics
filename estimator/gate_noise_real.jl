# The AR node's gate on the real cook: from a wrong starting noise and a
# weak prior, the alternation (fit the physics at the current whitening,
# the AR node on the residuals, refit) has to recover the calibration's
# noise (sigma_eff 1.54 K / rho 0.83 on the pit, 1.70 / 0.947 on the
# core) and the calibration's MAP; with the calibration's own posterior
# as the prior it has to stay there.
#
# Run:  julia --project=estimator estimator/gate_noise_real.jl
# Writes data/calibration/estimator_gate_noise_result.json.
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Printf

REPO = normpath(joinpath(@__DIR__, ".."))
gate_path = joinpath(REPO, "data", "calibration", "estimator_gate_real.json")
d = load_gate(gate_path)
raw = JSON.parsefile(gate_path)
nz = raw["noise"]
floor = Float64(nz["sigma_floor_K"])
mask_pit = Bool.(raw["mask_pit"]); mask_core = Bool.(raw["mask_core"])
obs_pit = Float64.(raw["obs_pit"]); obs_core = Float64.(raw["obs_core"])
sd_map = sqrt.(diag(d.cov_map))
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))

function run(label, σp, ρp, σc, ρc, priors)
    println("=== ", label)
    chs = [PitProbe(σp, ρp, mask_pit; data=obs_pit), MeatProbe(σc, ρc, mask_core; s=0.0, data=obs_core)]
    t0 = time()
    res = solve_map_channels(chs, d.prior, d; c0=d.z0, noise=priors, noise_rounds=4, verbose=true)
    dz = (res.z .- d.z_map) ./ sd_map
    @printf("  MAP vs calibration: max |dz|/sd %.3f (%s), median %.3f; F %.4f (calibration %.4f); %.0f s\n",
            maximum(abs.(dz)), d.names[argmax(abs.(dz))], sort(abs.(dz))[8], res.F, d.F_map, time() - t0)
    for (i, nm) in ((1, "pit"), (2, "core"))
        np = res.noise[i]
        ref_σ, ref_ρ = nz["sigma_$(nm)_eff"], nz["rho_$(nm)"]
        @printf("  %-4s σ %.3f ± %.3f (calibration %.3f)   ρ %.4f ± %.4f (calibration %.4f)\n",
                nm, np.σ, np.σ_sd, ref_σ, np.ρ, np.ρ_sd, ref_ρ)
    end
    mt = res.misfit.total
    @printf("  misfit: residual power/row %.3f, z_chi2 %.2f, z_rho1 %.2f; per channel %s\n", mt.ss / mt.n, mt.z_chi2, mt.z_rho1,
            join(["$(c.name) $(round(c.z_chi2; digits=2))/$(round(c.z_rho1; digits=2))" for c in res.misfit.channels], ", "))
    flush(stdout)
    return Dict("label" => label, "z" => res.z, "dz_sd" => dz, "F" => res.F,
                "noise" => Dict(string(i) => Dict(pairs(np)) for (i, np) in res.noise),
                "misfit_total" => Dict(pairs(mt)))
end

weak = Dict(1 => NoisePrior(3.0, 3.0, 0.5, 0.4; floor=floor), 2 => NoisePrior(3.0, 3.0, 0.5, 0.4; floor=floor))
r1 = run("wrong start (σ 3.0, ρ 0.5), weak prior", 3.0, 0.5, 3.0, 0.5, weak)
calp = Dict(1 => cal["pit"], 2 => cal["core"])
r2 = run("calibration prior, started at its means", cal["pit"].σ_mean, cal["pit"].ρ_mean, cal["core"].σ_mean, cal["core"].ρ_mean, calp)
r3 = run("fixed calibration noise (reference, no AR node)", nz["sigma_pit_eff"], nz["rho_pit"], nz["sigma_core_eff"], nz["rho_core"], nothing)

dst = joinpath(REPO, "data", "calibration", "estimator_gate_noise_result.json")
open(dst, "w") do io
    JSON.print(io, Dict("runs" => [r1, r2, r3], "calibration_noise" => nz, "names" => d.names))
end
println("wrote ", dst)
