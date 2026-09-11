# the filter beside the batch estimator on the constant real-cook mock: with no process noise the two should
# agree to within the hand-off approximation; prints both per window against the truth
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator, BrisketDynamics, JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
mock = length(ARGS) >= 1 ? ARGS[1] : "mock_realcook.json"
to_h = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 6.0
m = JSON.parsefile(joinpath(REPO, "data", "mock", mock))
gate = JSON.parsefile(joinpath(REPO, "data", "calibration", "prior_next_cook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"])); cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
prior_phys = [PriorSite(String(p["family"]), Float64(p["p1"]), Float64(p["p2"])) for p in gate["prior"]]
names = String.(gate["names"])
L_med = exp(prior_phys[4].p1)
extra = [LatentSpec("A_meat", "lognormal", log(cfg.A_meat), 0.3), LatentSpec("core_depth", "lognormal", log(0.8 * L_med), 0.5)]
sites = vcat(prior_phys, [e.site for e in extra]); allnames = vcat(names, [e.name for e in extra])
AREA, DEPTH = length(sites) - 1, length(sites)
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
cal = noise_priors_from_file(joinpath(REPO, "data", "calibration", "noise_prior_real.json"))
tostep(np, from_dt) = (a = dt / from_dt; NoisePrior(np.σ_mean, np.σ_sd, np.ρ_mean^a, np.ρ_sd * a * np.ρ_mean^(a - 1); floor=np.floor))
np = Dict(1 => tostep(cal["pit"], 600.0), 2 => tostep(cal["core"], 600.0), 3 => NoisePrior(100.0, 100.0, 0.8^(dt / 600), 0.3; floor=60.0), 4 => NoisePrior(0.02, 0.02, 0.5^(dt / 1800), 0.3; floor=0.015))
nc = m["noise"]; rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
edges = 3600.0 .* collect(2:2:to_h)
# the truth per window: constant, or by the hour for the drifting mock
x_true = Float64.(m["truth"]["x"])
truth_at(te) = haskey(m, "truth_by_hour") ? (hs = Float64.(m["truth_by_hour"]["hours"]); k = max(1, findlast(hs .<= te / 3600 + 1e-9)); Float64.(m["truth_by_hour"]["x"][k])) : x_true
z_of(x) = [isbeta(p) ? log(v / (1 - v)) : log(v) for (v, p) in zip(x, sites)]
q = zeros(length(sites))
for (nm, sd) in get(m, "drift", Dict("sd_per_hour" => Dict()))["sd_per_hour"]
    q[findfirst(==(nm), allnames)] = sd^2
end
println("mock ", mock, "; edges ", edges ./ 3600, "; process noise rates ", Dict(allnames[i] => q[i] for i in findall(q .> 0)))
t0 = time()
fr = filter_fit(readings, masks, t, u, pit, cfg, sites, names, events, edges; q, noise=np, cap=Float64(nc["hum_cap"]), depth_latent=DEPTH, area_latent=AREA, verbose=true)
tf = time() - t0
chs = [PitProbe(np[1].σ_mean, np[1].ρ_mean, masks.pit; data=readings.pit), MeatProbe(np[2].σ_mean, np[2].ρ_mean, masks.core; depth_latent=DEPTH, data=readings.core),
       CensoredHumidity(np[3].σ_mean, np[3].ρ_mean, masks.humidity; cap=Float64(nc["hum_cap"]), data=readings.humidity), MassScale(np[4].σ_mean, np[4].ρ_mean, masks.mass; data=readings.mass)]
d = estimator_setup(t, u, pit, cfg, prior_phys, names)
t0 = time()
br = stream_fit(chs, sites, d, edges; physics=Physics1D(area_latent=AREA, events=events), noise=np, noise_rounds=1, levelled=true, verbose=false)
tb = time() - t0
@printf("filter %.0f s, batch %.0f s\n", tf, tb)
println("window   filter: median|err|/sd  max (site)  |  batch: median  max (site)  |  filter vs batch: max |Δz|/sd_batch")
for (f, b) in zip(fr, br)
    zt = z_of(truth_at(f.t))
    ef = abs.((f.z .- zt) ./ f.sd); eb = abs.((b.z .- zt) ./ b.sd)
    @printf("%5.1f h  %6.2f  %5.2f (%s)  |  %6.2f  %5.2f (%s)  |  %5.2f\n", f.t / 3600, sort(ef)[div(end + 1, 2)], maximum(ef), allnames[argmax(ef)],
            sort(eb)[div(end + 1, 2)], maximum(eb), allnames[argmax(eb)], maximum(abs.(f.z .- b.z) ./ b.sd))
end
for nm in ("u_leak", "k_e", "h_c", "r_max", "a0", "C_th")
    i = findfirst(==(nm), allnames)
    println(rpad(nm, 8), " truth ", join([@sprintf("%+.2f", z_of(truth_at(f.t))[i]) for f in fr], " "), " | filter ", join([@sprintf("%+.2f±%.2f", f.z[i], f.sd[i]) for f in fr], " "), " | batch ", join([@sprintf("%+.2f±%.2f", b.z[i], b.sd[i]) for b in br], " "))
end
