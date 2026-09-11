# per coverage cook: how much of the humidity channel is censored by 14 h, against the a_myosin error at 14 h
include(joinpath(@__DIR__, "particle_probe_setup.jl"))   # the bench setup (seed 1 fit happens there; harmless)
using JSON
recs_json = vcat([JSON.parsefile(joinpath(REPO, "data", "mock", f))["cooks"] for f in ("coverage_realcook_1.json", "coverage_realcook_8.json", "coverage_realcook_15.json")]...)
ia = findfirst(==("a_myosin"), allnames); ik = findfirst(==("k_e"), allnames)
println("seed  censored%  hum_mean_Pa  a_myosin err/sd  k_e err/sd  W_serve")
for c in recs_json
    sd = c["seed"]; rng = MersenneTwister(1000 + sd)
    zt = draw_truth(rng, prior); xt = from_z(zt)
    θ, φ, A, depth = xt[1:8], xt[9:15], xt[16], xt[17]
    o = monolith_solve(θ, φ, u, t, t; pit, cfg, A_meat=A, events...)
    readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=θ[4], depth=depth)
    keep = masks.humidity .& (t .<= 14 * 3600)
    cens = count(readings.humidity[keep] .>= nc.hum_cap - 1e-9) / max(count(keep), 1)
    e = c["batches"][end]["err_sd"]
    @printf("%4d  %8.2f  %11.0f  %15.2f  %10.2f\n", sd, cens, sum(clean.humidity[keep]) / max(count(keep), 1), e[ia], e[ik])
end
