# A/B of the dynamics factors' relinearization tolerances on the real-cook mock's first arrivals: the exact policy
# (latents and node within 1e-3) against a loosened one, at a given lag; per arrival the wall, iterations and
# Jacobians, and the beliefs' agreement (latents in posterior sds, the temperatures in K, the widths' ratio).
# Run: julia --project=estimator ab_relin.jl <arrivals> <lag> <relin_tol_B> <relin_tol_state_B>
using Pkg
const EST = joinpath(homedir(), ".julia", "dev", "BrisketDynamics", "estimator")
Pkg.activate(EST)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(EST, "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf, Statistics
const E = BrisketEstimator
REPO = normpath(joinpath(EST, ".."))
K_END = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
LAG = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
RT_B = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.02
RTS_B = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.05
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
ev = m["events"]
events = [CookEvent(:wrap, :brisket, ev["t_wrap"]), CookEvent(:pull, :brisket, ev["t_pull"]), CookEvent(:add, :pit, ev["t_add"]),
          CookEvent(:spritz, :brisket, ev["t_spritz"]; dt=ev["dt_spritz"], m=ev["m_spritz"])]
nc = RealCookNoise()
sys = CookSystem(Slab1D(:brisket; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[:brisket])
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=nc.pit_sd, tau=nc.pit_tau)),
           SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=nc.core_sd, tau=nc.core_tau)),
           SensorSpec(:humidity, ChamberVapor(), NoiseModel(white_sd=10.0, ar_sd=nc.hum_sd, tau=nc.hum_tau, cap=nc.hum_cap)),
           SensorSpec(:mass, TotalMass(), NoiseModel(white_sd=0.005, ar_sd=nc.mass_sd, tau=nc.mass_tau))]
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
recs = MeasurementRecord[]
for k in 1:K_END, id in (:pit, :core, :humidity, :mass)
    getfield(masks, id)[k] && push!(recs, MeasurementRecord(id, t[k], getfield(readings, id)[k]))
end
arrivals = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
clog = CookLog(t, u; events, sensors, measurements=recs)
model = SystemModel(sys, clog, blk; sn=StateNoise())

function run_policy(label; relin_tol, relin_tol_state)
    sm = online_init(model; lag=LAG, relin_tol, relin_tol_state)
    seen = IdDict{Any,Int}()
    out = NamedTuple[]
    for grp in arrivals
        r = assimilate!(sm, grp)
        nj = 0
        for f in values(sm.dyn); j0 = get(seen, f, 0); nj += f.n_jacobians - j0; seen[f] = f.n_jacobians; end
        push!(out, (t=r.t, wall=r.wall, iters=r.iters, converged=r.converged, committed=r.committed, jac=nj, gradient=r.gradient,
                    newton=r.newton, newton_sd=r.newton_sd, z=copy(r.z), zsd=sqrt.(diag(r.z_cov)), Tpit=r.T_pit, Tc=r.pieces[:brisket].T_centre, Ts=r.pieces[:brisket].T_surface, nis=r.nis))
    end
    @printf("%s: %d arrivals, wall %.0f s (mean %.1f, max %.0f), iterations mean %.1f max %d, Jacobians %d (mean %.1f), not committed %d\n", label, length(out),
            sum(r.wall for r in out), mean(r.wall for r in out), maximum(r.wall for r in out), mean(r.iters for r in out), maximum(r.iters for r in out),
            sum(r.jac for r in out), mean(r.jac for r in out), count(r -> !r.committed, out))
    flush(stdout)
    return out
end
println("A/B at lag ", LAG, ": A exact (1e-3, 1e-3) against B (", RT_B, ", ", RTS_B, "); ", length(arrivals), " arrivals")
flush(stdout)
A = run_policy("A"; relin_tol=1e-3, relin_tol_state=1e-3)
B = run_policy("B"; relin_tol=RT_B, relin_tol_state=RTS_B)
dz = maximum(maximum(abs.(a.z .- b.z) ./ a.zsd) for (a, b) in zip(A, B))
dsd = maximum(maximum(abs.(b.zsd ./ a.zsd .- 1)) for (a, b) in zip(A, B))
dT = maximum(max(abs(a.Tpit[1] - b.Tpit[1]), abs(a.Tc[1] - b.Tc[1]), abs(a.Ts[1] - b.Ts[1])) for (a, b) in zip(A, B))
dTsd = maximum(max(abs(a.Tpit[2] / b.Tpit[2] - 1), abs(a.Tc[2] / b.Tc[2] - 1)) for (a, b) in zip(A, B))
dnis = maximum(abs(a.nis - b.nis) for (a, b) in zip(A, B) if isfinite(a.nis) && isfinite(b.nis))
@printf("B against A: latents within %.2e posterior sd, widths within %.2e relative; temperatures within %.2e K, their sds within %.2e relative; NIS within %.2e\n", dz, dsd, dT, dTsd, dnis)
@printf("gradients at the accepted points: A max %.2e, B max %.2e\n", maximum(r.gradient for r in A), maximum(r.gradient for r in B))
open(joinpath(@__DIR__, "ab_relin_lag$(LAG).json"), "w") do io
    JSON.print(io, Dict("lag" => LAG, "B" => Dict("relin_tol" => RT_B, "relin_tol_state" => RTS_B),
                        "A_rows" => [Dict("t" => r.t, "wall" => r.wall, "iters" => r.iters, "jac" => r.jac, "gradient" => r.gradient) for r in A],
                        "B_rows" => [Dict("t" => r.t, "wall" => r.wall, "iters" => r.iters, "jac" => r.jac, "gradient" => r.gradient) for r in B],
                        "dz" => dz, "dsd" => dsd, "dT" => dT, "dTsd" => dTsd, "dnis" => dnis), 1)
end
println("wrote ab_relin_lag$(LAG).json")
println("per arrival: t [min] | A gradient, Newton step (scaled), Newton step [posterior sd] | B the same | the latents' gap B-A [posterior sd] | the temperatures' gap [K]")
for (a, b) in zip(A, B)
    @printf("  %6.1f | %.1e %.1e %6.2f | %.1e %.1e %6.2f | %5.2f | %5.2f\n", a.t / 60, a.gradient, a.newton, a.newton_sd, b.gradient, b.newton, b.newton_sd,
            maximum(abs.(a.z .- b.z) ./ a.zsd), max(abs(a.Tpit[1] - b.Tpit[1]), abs(a.Tc[1] - b.Tc[1]), abs(a.Ts[1] - b.Ts[1])))
end
@printf("the Newton step at the accepted points: in posterior sds A max %.2f, B max %.2f; in scaled coordinates A max %.2e, B max %.2e (the diagonal certificate's tolerance 1e-4)\n",
        maximum(r.newton_sd for r in A), maximum(r.newton_sd for r in B), maximum(r.newton for r in A), maximum(r.newton for r in B))
