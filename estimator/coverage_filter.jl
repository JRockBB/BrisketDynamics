# The coverage and consistency test of the filter on the graph (sequential.jl) on the DRIFTING plant: the same
# nine cooks as coverage_drift.jl (truths drawn from the pilot's prior, walked by drift.jl, read by the real-cook
# instruments with their own noise), the filter run update by update as the samples arrive to the last edge, its
# belief of the latents scored against the truth in force at every batch edge (the same edges as the batch and the
# smoother, so the estimators pool side by side), its belief of the pit, centre and surface temperatures scored
# against the plant's at every update, and the consistency tests at every update: the normalized innovation
# squared with its degrees of freedom, the normalized estimation error squared of the latents and of the three
# temperatures, the standardized innovations per channel. Writes data/mock/<out.json> (coverage_filter_<seed0>.json
# by default), after every cook; plot_coverage.py reads it with the key err_sd_graphfilter, plot_filter_consistency.py
# the consistency tests.
#
# Run:  julia --project=estimator estimator/coverage_filter.jl [prior.json] [n_cooks] [seed0] [out.json] [edges_to_h] [adapt_level 0/1] [state_noise.json|default]
#   adapt_level: 0 (the default) keeps every channel's level at one while the state's process noise is earned from this
#   bench (a level reads the belief's consistency and cannot be judged before the belief is consistent); 1 lets it follow
#   state_noise.json (data/calibration): the state's process noise and initial widths the filter runs with (the file the
#   bench driver estimator/bench_state_noise.py writes; the StateNoise defaults when there is none, or with "default")
# The record scores the belief's state group by group (the normalized estimation error squared of each field and
# scalar block against the plant's state, per update): the bench driver reads it to earn the rates.
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
adapt_level = length(ARGS) >= 6 ? ARGS[6] == "1" : false
sn_file = length(ARGS) >= 7 ? ARGS[7] : "state_noise.json"
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", prior_file), cfg)
sites, allnames, idx = blk.sites, blk.names, blk.idx
AREA, DEPTH = blk.area, blk.depth
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
edges = unique(vcat(3600.0 .* collect(2:2:floor(Int, edges_to_h)), [3600.0 * edges_to_h]))     # every two hours, and the run's end
nc = RealCookNoise()
np0 = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
noise = [np0[i] for i in 1:4]
dr = drift_rates(blk.file)     # the rates the filter is told (the plant walks with drift.jl's own; the same by default: the test is of the mechanism)
q = zeros(length(sites)); for (nm, sd) in dr.sd_per_hour; q[idx[nm]] = sd^2; end
jumps = [(idx[nm], events.t_add, sd^2) for (nm, sd) in dr.jump_log_sd]
sn = sn_file == "default" ? StateNoise() : load_state_noise(isabspath(sn_file) ? sn_file : joinpath(REPO, "data", "calibration", sn_file))
fingerprint = physics_fingerprint()
group_dof = Dict{String,Int}()          # each state group's degrees of freedom (filled from the first cook's layout)

cooks = Any[]; skipped = Int[]
dst = joinpath(REPO, "data", "mock", out_name)
save() = open(dst, "w") do io
    JSON.print(io, Dict("plant" => "monolith (1-D) in segments, parameters drifting (drift.jl); real-cook instruments", "names" => allnames,
                        "prior_file" => prior_file, "edges_h" => edges ./ 3600, "cooks" => cooks, "skipped_seeds" => skipped,
                        "drift" => Dict("sd_per_hour" => DRIFT_SD_PER_HOUR, "jump_log_sd" => JUMP_LOG_SD), "q" => q,
                        "state_noise" => Dict(string(k) => getfield(sn, k) for k in fieldnames(StateNoise)), "state_noise_file" => sn_file,
                        "physics" => fingerprint, "adapt_level" => adapt_level, "groups" => collect(STATE_GROUPS), "group_dof" => group_dof))
end
println("the graph filter: prior ", prior_file, "; ", n_cooks, " cooks from seed ", seed0, "; edges to ", edges_to_h, " h; updates on arrival; the level ",
        adapt_level ? "following the innovations" : "fixed at one (the bench earns the state's process noise)", "; the state noise from ", sn_file,
        "; physics ", fingerprint, "; record ", out_name)
println("  rates [units/sqrt(h)]: ", join(["$g $(getfield(sn, Symbol(g)))" for g in STATE_GROUPS], ", "))
println("  initial sds: ", join(["$g $(getfield(sn, Symbol(g * "0")))" for g in STATE_GROUPS], ", "))
flush(stdout)
for k in 1:n_cooks
    seed = seed0 + k - 1
    rng = MersenneTwister(2000 + seed)
    z0 = draw_truth(rng, sites; depth=DEPTH)
    dd = DriftDraw(rng, idx)
    seg_edges, blocks = drift_segments(t, z0, dd, events.t_add)
    xs = [from_z(z, sites) for z in blocks]
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
    chs = realcook_channels(readings, masks, np0; cap=nc.hum_cap, depth_latent=DEPTH)
    recs = try
        sequential_filter(chs, t, u, pit, cfg, sites, allnames, events; noise, area_latent=AREA, q, jumps, sn, adapt_level, diagnostics=true, verbose=false, t_end=edges[end])
    catch err
        err isa InterruptException && rethrow()
        @printf("cook %d (seed %d): the filter failed (%s); skipped\n", k, seed, sprint(showerror, err)[1:min(end, 100)]); push!(skipped, seed); save(); continue
    end
    wall = time() - t0
    # the latents at the batch edges: the update nearest the edge
    at_edge(te) = recs[argmin(abs.([r.t for r in recs] .- te))]
    batches = [(r = at_edge(te); zt = truth_at(z0, dd, te, events.t_add);
                Dict("t_h" => te / 3600, "z" => r.z, "sd" => r.z_sd, "err_sd" => (r.z .- zt) ./ r.z_sd, "z_true" => zt,
                     "err_sd_graphfilter" => (r.z .- zt) ./ r.z_sd)) for te in edges]
    # the temperatures at every update against the plant's, and the consistency tests: the normalized innovation
    # squared of the update (chi-squared with `dof` when the filter is consistent), the normalized estimation error
    # squared of the latents against the truth in force (chi-squared with 17, or 16 without a0) and of the three
    # temperatures jointly (chi-squared with 3), and the standardized innovations per channel for the whiteness test
    kk(te) = searchsortedlast(t, te + 1e-9)
    ia0 = idx["a0"]; keep16 = [i for i in 1:length(sites) if i != ia0]
    function nees(e, C)
        F = cholesky(Matrix(Symmetric(C)); check=false)
        issuccess(F) ? dot(e, F \ e) : NaN
    end
    # the state by group: the error against the plant's state at the update's grid point, in the block's correlation
    # form (a field block's entries span orders of magnitude), as NEES per degree of freedom
    lay = FilterLayout(size(o.T_field, 1), length(sites), 4)
    groups = state_groups(lay)
    gdof = group_dofs(lay, sn)
    for (g, _) in groups; group_dof[g] = gdof[g]; end
    function nees_corr(e, C)
        s = sqrt.(max.(diag(C), 1e-300))
        F = cholesky(Matrix(Symmetric(C ./ (s * s'))); check=false)
        issuccess(F) ? dot(e ./ s, F \ (e ./ s)) : NaN
    end
    # a field under :offset carries one draw for its nodes: it is scored on its bulk, the node mean, against the
    # plant's, one degree of freedom (its shape is the physics', not the noise's); otherwise node by node
    function group_nees(r)
        ics, a = handoff(o, r.k)
        ex = r.x .- pack_state(ics, a, lay)                    # its own name: an `e` here would be the loop's `e`, shared through the closure
        Dict(g => (gdof[g] == 1 && length(rg) > 1 ? (v = sum(r.x_cov[rg, rg]) / length(rg)^2; sum(ex[rg]) / length(rg))^2 / v :
                   nees_corr(ex[rg], r.x_cov[rg, rg]) / length(rg)) for (g, rg) in groups)
    end
    temps = [(zt = truth_at(z0, dd, r.t, events.t_add); ez = r.z .- zt;
              eT = [r.T_pit[1] - o.Tpit[kk(r.t)], r.T_centre[1] - o.Tcore[kk(r.t)], r.T_surface[1] - o.Tsurf[kk(r.t)]];
              Dict("t_h" => r.t / 3600, "k" => r.k, "updated" => r.updated, "nees_g" => group_nees(r),
                   "pit" => [r.T_pit[1], r.T_pit[2], o.Tpit[kk(r.t)]], "centre" => [r.T_centre[1], r.T_centre[2], o.Tcore[kk(r.t)]],
                   "surface" => [r.T_surface[1], r.T_surface[2], o.Tsurf[kk(r.t)]],
                   "nis" => r.nis, "dof" => r.dof, "innov" => Dict(string(ci) => v for (ci, v) in r.innov),
                   "nees_z" => nees(ez, r.z_cov), "nees_z16" => nees(ez[keep16], r.z_cov[keep16, keep16]), "nees_T" => nees(eT, r.T_cov),
                   "level" => r.level, "faulty" => r.faulty, "noise" => [[v.σ, v.ρ] for v in r.noise], "wall_s" => r.wall)) for r in recs]
    push!(cooks, Dict("seed" => seed, "z_by_hour" => [truth_at(z0, dd, h * 3600.0, events.t_add; from=true) for h in 0:29], "batches" => batches,
                      "temps" => temps, "wall_filter_s" => wall, "n_updates" => length(recs)))
    e = abs.(batches[end]["err_sd"])
    ep = [(r["pit"][1] - r["pit"][3]) / r["pit"][2] for r in temps]; ec = [(r["centre"][1] - r["centre"][3]) / r["centre"][2] for r in temps]; es = [(r["surface"][1] - r["surface"][3]) / r["surface"][2] for r in temps]
    nis_r = [r["nis"] / max(r["dof"], 1) for r in temps if r["dof"] > 0]
    @printf("cook %d (seed %d): at %.0f h |err|/sd max %.2f (%s) median %.2f; the temperatures' |err|/sd median pit %.2f centre %.2f surface %.2f; NIS/dof mean %.2f, NEES/16 mean %.2f, NEES_T/3 mean %.2f; %d updates, %.0f s\n",
            k, seed, edges[end] / 3600, maximum(e), allnames[argmax(e)], sort(e)[div(end + 1, 2)],
            sort(abs.(ep))[div(end + 1, 2)], sort(abs.(ec))[div(end + 1, 2)], sort(abs.(es))[div(end + 1, 2)],
            sum(nis_r) / length(nis_r), sum(r["nees_z16"] for r in temps) / 16 / length(temps), sum(r["nees_T"] for r in temps) / 3 / length(temps), length(recs), wall)
    for nm in ("u_leak", "k_e", "h_c", "r_max")
        i = idx[nm]
        @printf("    %-7s truth %s | filter %s\n", nm, join([@sprintf("%+.2f", b["z_true"][i]) for b in batches], " "), join([@sprintf("%+.2f", b["err_sd"][i]) for b in batches], " "))
    end
    gm(g) = (v = [r["nees_g"][g] for r in temps if isfinite(r["nees_g"][g])]; isempty(v) ? NaN : sum(v) / length(v))
    println("    the state by group, NEES/dof mean: ", join([@sprintf("%s %.2f", g, gm(g)) for g in STATE_GROUPS], ", "))
    flush(stdout)
    save()
end
println("wrote ", dst, " (", length(cooks), " cooks, ", length(skipped), " skipped)")
