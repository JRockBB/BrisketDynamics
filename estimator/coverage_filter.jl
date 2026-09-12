# The coverage test of the sequential filter on the graph (sequential.jl) on the DRIFTING plant: the same nine
# cooks as coverage_drift.jl (truths drawn from the pilot's prior, walked by drift.jl, read by the real-cook
# instruments with their own noise), the filter run slice by slice to the last edge, its belief of the latents
# scored against the truth in force at every batch edge (the same edges as the batch and the smoother, so the
# three pool side by side), and its belief of the pit, centre and surface temperatures scored against the
# plant's at every slice. Writes data/mock/<out.json> (coverage_filter_<seed0>.json by default), after every
# cook; plot_coverage.py reads it with the key err_sd_graphfilter.
#
# Run:  julia --project=estimator estimator/coverage_filter.jl [prior.json] [n_cooks] [seed0] [out.json] [edges_to_h] [step_min]
using Pkg
Pkg.activate(@__DIR__)
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, ".."))
prior_file = length(ARGS) >= 1 ? ARGS[1] : "prior_next_cook.json"
n_cooks = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
seed0 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
out_name = length(ARGS) >= 4 ? ARGS[4] : "coverage_filter_$(seed0).json"
edges_to_h = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 14.0
step_min = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 10.0
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
L_med = exp(prior_phys[4].p1)
sites = vcat(prior_phys, [PriorSite("lognormal", log(cfg.A_meat), 0.3), PriorSite("lognormal", log(0.8 * L_med), 0.5)])
allnames = vcat(names, ["A_meat", "core_depth"]); idx = Dict(n => i for (i, n) in enumerate(allnames))
AREA, DEPTH = length(sites) - 1, length(sites)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
edges = 3600.0 .* collect(2:2:floor(Int, edges_to_h))
nc = RealCookNoise()
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
np0 = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0), 3 => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), 4 => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
q = zeros(length(sites)); for (nm, sd) in DRIFT_SD_PER_HOUR; q[idx[nm]] = sd^2; end
jumps = [(idx[nm], events.t_add, sd^2) for (nm, sd) in JUMP_LOG_SD]

function beta_rand(rng, a, b)
    x = range(1e-6, 1 - 1e-6; length=4001); lp = (a - 1) .* log.(x) .+ (b - 1) .* log.(1 .- x); p = exp.(lp .- maximum(lp)); c = cumsum(p); c ./= c[end]
    r = rand(rng); k = clamp(searchsortedfirst(c, r), 2, length(c)); x[k - 1] + (x[k] - x[k - 1]) * (r - c[k - 1]) / max(c[k] - c[k - 1], 1e-300)
end
function draw_truth(rng)
    z = [isbeta(p) ? (x = beta_rand(rng, p.p1, p.p2); log(x / (1 - x))) : p.p1 + p.p2 * randn(rng) for p in sites]
    z[DEPTH] = min(z[DEPTH], log(0.9 * exp(z[4]))); z
end
from_z_all(z) = [isbeta(p) ? 1 / (1 + exp(-v)) : exp(v) for (v, p) in zip(z, sites)]

cooks = Any[]; skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    JSON.print(io, Dict("plant" => "monolith (1-D) in segments, parameters drifting (drift.jl); real-cook instruments", "names" => allnames,
                        "prior_file" => prior_file, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped, "step_min" => step_min,
                        "drift" => Dict("sd_per_hour" => DRIFT_SD_PER_HOUR, "jump_log_sd" => JUMP_LOG_SD), "q" => q))
end
println("the graph filter: prior ", prior_file, "; ", n_cooks, " cooks from seed ", seed0, "; edges to ", edges_to_h, " h; slices of ", step_min, " min; record ", out_name)
flush(stdout)
for k in 1:n_cooks
    seed = seed0 + k - 1
    rng = MersenneTwister(2000 + seed)
    z0 = draw_truth(rng)
    dd = DriftDraw(rng, idx)
    seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
    xs = [from_z_all(z) for z in blocks]
    A, depth = xs[1][AREA], xs[1][DEPTH]
    t0 = time()
    o = try
        solve_segments([x[1:8] for x in xs], [x[9:15] for x in xs], u, t, seg_edges; pit, cfg, A_meat=A, events...)
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the plant failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 80)]); push!(skipped, seed); save(); continue
    end
    all(isfinite, o.Tcore) || (@printf("cook %d (seed %d): the plant is not finite; skipped\n", k, seed); push!(skipped, seed); save(); continue)
    readings, clean, masks, lid = realcook_readings(o, t, events, nc, rng; A=A, L=xs[1][4], depth=depth)
    rd = (pit=Float64.(readings.pit), core=Float64.(readings.core), humidity=Float64.(readings.humidity), mass=Float64.(readings.mass))
    mk = (pit=Bool.(masks.pit), core=Bool.(masks.core), humidity=Bool.(masks.humidity), mass=Bool.(masks.mass))
    recs = try
        sequential_filter(rd, mk, t, u, pit, cfg, sites, names, events; step=step_min * 60, noise=deepcopy(np0), cap=nc.hum_cap,
                          depth_latent=DEPTH, area_latent=AREA, q, jumps, diagnostics=true, verbose=false, t_end=edges[end])
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the filter failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 100)]); push!(skipped, seed); save(); continue
    end
    wall = time() - t0
    # the latents at the batch edges: the slice ending at the edge
    at_edge(te) = recs[argmin(abs.([r.t for r in recs] .- te))]
    batches = [(r = at_edge(te); zt = truth_at(z0, dd, te, events.t_add);
                Dict("t_h" => te / 3600, "z" => r.z, "sd" => r.z_sd, "err_sd" => (r.z .- zt) ./ r.z_sd, "z_true" => zt,
                     "err_sd_graphfilter" => (r.z .- zt) ./ r.z_sd)) for te in edges]
    # the temperatures at every slice against the plant's, and the consistency tests: the normalized innovation
    # squared of the slice (chi-squared with `dof` when the filter is consistent), the normalized estimation error
    # squared of the latents against the truth in force (chi-squared with 17, or 16 without a0) and of the three
    # temperatures jointly (chi-squared with 3), and the standardized innovations per channel for the whiteness test
    kk(te) = searchsortedlast(t, te + 1e-9)
    ia0 = idx["a0"]; keep16 = [i for i in 1:length(sites) if i != ia0]
    function nees(e, C)
        F = cholesky(Matrix(Symmetric(C)); check=false)
        issuccess(F) ? dot(e, F \ e) : NaN
    end
    temps = [(zt = truth_at(z0, dd, r.t, events.t_add); e = r.z .- zt;
              eT = [r.T_pit[1] - o.Tpit[kk(r.t)], r.T_centre[1] - o.Tcore[kk(r.t)], r.T_surface[1] - o.Tsurf[kk(r.t)]];
              Dict("t_h" => r.t / 3600,
                   "pit" => [r.T_pit[1], r.T_pit[2], o.Tpit[kk(r.t)]], "centre" => [r.T_centre[1], r.T_centre[2], o.Tcore[kk(r.t)]],
                   "surface" => [r.T_surface[1], r.T_surface[2], o.Tsurf[kk(r.t)]],
                   "nis" => r.nis, "dof" => r.dof, "innov" => Dict(string(ci) => v for (ci, v) in r.innov),
                   "nees_z" => nees(e, r.z_cov), "nees_z16" => nees(e[keep16], r.z_cov[keep16, keep16]), "nees_T" => nees(eT, r.T_cov))) for r in recs]
    push!(cooks, Dict("seed" => seed, "z_by_hour" => [truth_at(z0, dd, h * 3600.0, events.t_add; from=true) for h in 0:29], "batches" => batches,
                      "temps" => temps, "wall_filter_s" => wall, "n_slices" => length(recs)))
    e = abs.(batches[end]["err_sd"])
    ep = [(r["pit"][1] - r["pit"][3]) / r["pit"][2] for r in temps]; ec = [(r["centre"][1] - r["centre"][3]) / r["centre"][2] for r in temps]; es = [(r["surface"][1] - r["surface"][3]) / r["surface"][2] for r in temps]
    nis_r = [r["nis"] / max(r["dof"], 1) for r in temps if r["dof"] > 0]
    @printf("cook %d (seed %d): at %.0f h |err|/sd max %.2f (%s) median %.2f; the temperatures' |err|/sd median pit %.2f centre %.2f surface %.2f; NIS/dof mean %.2f, NEES/16 mean %.2f, NEES_T/3 mean %.2f; %d slices, %.0f s\n",
            k, seed, edges[end] / 3600, maximum(e), allnames[argmax(e)], sort(e)[div(end + 1, 2)],
            sort(abs.(ep))[div(end + 1, 2)], sort(abs.(ec))[div(end + 1, 2)], sort(abs.(es))[div(end + 1, 2)],
            sum(nis_r) / length(nis_r), sum(r["nees_z16"] for r in temps) / 16 / length(temps), sum(r["nees_T"] for r in temps) / 3 / length(temps), length(recs), wall)
    for nm in ("u_leak", "k_e", "h_c", "r_max")
        i = idx[nm]
        @printf("    %-7s truth %s | filter %s\n", nm, join([@sprintf("%+.2f", b["z_true"][i]) for b in batches], " "), join([@sprintf("%+.2f", b["err_sd"][i]) for b in batches], " "))
    end
    flush(stdout)
    save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
