using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "src", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_1d.json")); gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "estimator_gate_real.json"))
rec = JSON.parsefile(joinpath(REPO, "data", "mock", "closed_loop_realcook.json"))
meta = m["meta"]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = read_sites(gate["prior"])
PLAN_GRID = collect(0.0:600.0:30 * 3600.0)
d = estimator_setup(PLAN_GRID, fill(0.15, length(PLAN_GRID)), pit, cfg, prior_phys, String.(gate["names"]))
z_true = Float64.(rec["z_true"])
goals = PlanGoals()
ex = rec["executed"]
for hour in (4, 8)
    ck = rec["checks"][hour + 1]
    z_hat = Float64.(ck["posterior_z"])
    s = Schedule(PLAN_GRID; u0=0.15)
    s.u .= [v === nothing ? NaN : Float64(v) for v in ex["u"]]; s.u[PLAN_GRID .> hour * 3600] .= NaN; s.t_now = hour * 3600.0
    f = free_set(s)
    e = [abs(x) for x in ck["err_sd"]]
    @printf("\nfrom the state at %d h (the executed damper to then; posterior median |err|/sd %.2f, max %.2f):\n", hour, sort(e)[div(end + 1, 2)], maximum(e))
    @printf("%-28s", "wrap [h] ->"); for w in (8.5, 10.0, 11.5, 13.0); @printf("%9.1f", w); end; println()
    for (lab, z) in (("J at the truth", z_true), ("J at the posterior mean", z_hat))
        @printf("%-28s", lab)
        for w in (8.5, 10.0, 11.5, 13.0)
            plan = (u=[0.35, 0.35, 0.2], t_wrap=w * 3600, t_pull=15.0 * 3600, t_serve=16.7 * 3600, spritz=(t=10.4 * 3600, dt=3600.0, m=0.05), t_add=11.6 * 3600)
            ζ = encode_plan(plan, f, s)
            y = plan_outputs_rh(z, ζ, f, s, d; goals)
            @printf("%9.3f", objective_terms(y, goals).J)
        end
        println()
    end
    for (lab, z) in (("truth", z_true), ("belief", z_hat))
        plan = (u=[0.35, 0.35, 0.2], t_wrap=8.5 * 3600, t_pull=15.0 * 3600, t_serve=16.7 * 3600, spritz=(t=10.4 * 3600, dt=3600.0, m=0.05), t_add=11.6 * 3600)
        y = plan_outputs_rh(z, encode_plan(plan, f, s), f, s, d; goals); T = objective_terms(y, goals)
        @printf("  wrap 8.5 h at the %s: collagen %.3f juiciness %.3f L* %.1f smoke %.2f -> tenderness %.3f appearance %.3f taste %.3f burn %.3f\n", lab, y[1], y[6], y[3], y[5], T.tenderness, T.appearance, T.taste, T.surf)
    end
end
