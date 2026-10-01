# Pooled consistency of stream_bench.jl records: per cook and over all cooks, the innovations' NIS per degree of freedom
# against its chi-squared band, the three temperatures' and the latents' NEES per dimension, every state group's NEES, the
# share of the truth inside the belief's 95 % band per temperature, the latents' standardized errors at the two-hour edges.
# Run: julia --project=estimator estimator/scripts/stream_score.jl data/mock/stream_full_seed1.json [more.json ...]
using JSON, Statistics, Printf
using SpecialFunctions: gamma_inc

files = ARGS
cooks = Any[]; names = nothing
for f in files
    d = JSON.parsefile(f)
    global names = d["names"]
    append!(cooks, d["cooks"])
end
isempty(cooks) && error("no cooks in $(files)")
function chi_q(p, d)
    lo, hi = 0.0, Float64(max(d, 1))
    while gamma_inc(d / 2, hi / 2)[1] < p; hi *= 2; end
    for _ in 1:80
        mid = (lo + hi) / 2
        gamma_inc(d / 2, mid / 2)[1] < p ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
chi_band(dof) = (chi_q(0.025, dof) / dof, chi_q(0.975, dof) / dof)

println("cooks: ", [c["seed"] for c in cooks], "   updates: ", sum(length(c["temps"]) for c in cooks), "   uncommitted: ", sum(c["n_uncommitted"] for c in cooks))
rows = [x for c in cooks for x in c["temps"]]
nis = sum(x["nis"] for x in rows if x["dof"] > 0 && x["nis"] !== nothing); dof = sum(x["dof"] for x in rows if x["dof"] > 0 && x["nis"] !== nothing)
lo, hi = chi_band(dof)
@printf("NIS/dof pooled %.3f   (a calibrated band %.3f to %.3f over %d dof; the updates are a sequence, so the band is a floor on the spread)\n", nis / dof, lo, hi, dof)
for key in ("pit", "centre", "surface")
    z = [(x[key][1] - x[key][3]) / x[key][2] for x in rows]
    @printf("  %-8s |err|/sd: rms %.2f, median |z| %.2f (0.67 calibrated), inside 95%% %.3f\n", key, sqrt(mean(z .^ 2)), median(abs.(z)), mean(abs.(z) .<= 1.96))
end
@printf("NEES_T/3 mean %.2f   NEES_z16/16 mean %.2f   (1 calibrated)\n", mean(x["nees_T"] for x in rows) / 3, mean(x["nees_z16"] for x in rows) / 16)
groups = sort(collect(keys(rows[1]["nees_g"])))
println("state groups, NEES/dof mean (median):")
for g in groups
    v = [x["nees_g"][g] for x in rows if x["nees_g"][g] !== nothing && isfinite(x["nees_g"][g])]
    @printf("  %-10s %.2f (%.2f)\n", g, mean(v), median(v))
end
println("latents at the edges, standardized error pooled over cooks (rms, mean), the worst:")
edges = [b["t_h"] for b in cooks[1]["batches"]]
for (j, te) in enumerate(edges)
    E = reduce(hcat, [Float64.(c["batches"][j]["err_sd"]) for c in cooks if length(c["batches"]) >= j])
    r = vec(sqrt.(mean(E .^ 2; dims=2)))
    w = sortperm(r; rev=true)[1:3]
    @printf("  %4.0f h: rms over all %.2f; worst %s\n", te, sqrt(mean(E .^ 2)), join(["$(names[i]) $(round(r[i]; digits=2))" for i in w], ", "))
end
for c in cooks
    rs = c["temps"]
    n = sum(x["nis"] for x in rs if x["dof"] > 0 && x["nis"] !== nothing); d = sum(x["dof"] for x in rs if x["dof"] > 0 && x["nis"] !== nothing)
    @printf("seed %d: NIS/dof %.3f, NEES_T/3 %.2f, NEES_z16/16 %.2f, %d updates, %.0f s\n", c["seed"], n / d, mean(x["nees_T"] for x in rs) / 3,
            mean(x["nees_z16"] for x in rs) / 16, length(rs), c["wall_filter_s"])
end
