# The finite-horizon observability of the filter's block on the real-cook mock: the open-loop observability matrix
# O = [H_1; H_2 F_2; H_3 F_3 F_2; ...] from the accumulator's kept pieces (each update's equivalent transition and its
# standardized rows), scaled by the initial belief's Cholesky factor so a direction's singular value says how many
# of its own sds it moves the standardized innovations over the horizon; the singular values, the rank at a
# threshold, and the weakest directions read in the block's coordinates (the latents by name, the temperature
# field's node under the core probe, the pit, each channel's discrepancy). The review of 2026-09-24 asked for it
# before the bench reruns: the reading is physics plus discrepancy, an instantaneous null direction only the
# dynamics can separate.
#   julia --project=estimator estimator/scripts/observability.jl [hours=2] [out.json]   (writes scratchpad/observability.json)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
hours = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 2.0
out = length(ARGS) >= 2 ? ARGS[2] : joinpath(REPO, "scratchpad", "observability.json")
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
sites, names = blk.sites, blk.names
AREA, DEPTH = blk.area, blk.depth
ev = m["events"]
events = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
nc = RealCookNoise()
np = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
channels() = realcook_channels(readings, masks, np; cap=nc.hum_cap, depth_latent=DEPTH)
noise = [np[i] for i in 1:4]
sn = load_state_noise(; strict=false, warn=false)
q = zeros(length(sites))

st = filter_init(channels(), t, u, pit, cfg, sites, names, events; noise, area_latent=AREA, q, sn, n_sigma=64)
lay = st.lay
P0 = Matrix(Symmetric(st.P))
L0 = cholesky(Symmetric(E._stable_cov(P0))).L                       # the initial belief's factor: directions in its own sds
st.als = ALSAccumulator(st.sn, st.lay; lag_s=3600.0, bin_s=dt, keep=true)
t0 = time()
recs = filter_advance!(st; t_to=hours * 3600, diagnostics=true)
pieces = st.als.pieces
@printf("the filter to %.1f h: %d updates in %.0f s; the block is %d wide (%d latents, %d state, %d discrepancies)\n", hours, length(recs), time() - t0, lay.n, lay.n_z, lay.n_state, lay.n_ch)
gaps = [r.gap for r in recs if r.updated]
@printf("the updates' fixed-point gaps: max %.4f, median %.4f sd\n", maximum(gaps), sort(gaps)[div(end + 1, 2)])

# the open-loop observability matrix over the horizon, in the initial belief's sds
Φ = Matrix{Float64}(I, lay.n, lay.n)
rows = Matrix{Float64}[]
labels = String[]
for (k, pc) in enumerate(pieces)
    global Φ = pc.F * Φ
    size(pc.H, 1) == 0 && continue
    push!(rows, pc.H * Φ * L0)
    append!(labels, ["$(round(pc.t / 3600; digits=2)) h ch$(c)" for c in pc.rowch])
end
O = reduce(vcat, rows)
U, S, V = svd(O)
n = lay.n
@printf("the observability matrix: %d rows over %d directions; singular values (in sds moved over the horizon):\n", size(O, 1), n)
println("  the ten largest: ", join([@sprintf("%.3g", s) for s in S[1:10]], " "))
println("  the ten smallest: ", join([@sprintf("%.2g", s) for s in S[max(1, end - 9):end]], " "))
for thr in (1.0, 0.1, 0.01, 1e-3)
    @printf("  directions moving the innovations by more than %.3g sd over %.1f h: %d of %d\n", thr, hours, count(>=(thr), S), n)
end

# the block's coordinate names, for reading the directions
xr = state_range(lay); dr = disc_range(lay)
groups = state_groups(lay)
coord = Vector{String}(undef, n)
for i in 1:lay.n_z; coord[i] = names[i]; end
for (g, rg) in groups, (j, i) in enumerate(rg); coord[xr[i]] = length(rg) > 1 ? "$(g)[$(j)]" : g; end
for c in 1:lay.n_ch; coord[dr[c]] = "disc$(c)"; end
core_node = centre_index(lay)                                          # the temperature field's node under the core probe

# per coordinate: how far a one-sd move of that coordinate alone moves the standardized innovations over the
# horizon (the norm of its column of the unwhitened observability matrix, scaled by its own sd); the latents by
# name, each state group by its most responsive node, each discrepancy
sd0 = sqrt.(max.(diag(P0), 1e-300))
Φ = Matrix{Float64}(I, lay.n, lay.n)
Oraw = Matrix{Float64}[]
for pc in pieces
    global Φ = pc.F * Φ
    size(pc.H, 1) == 0 && continue
    push!(Oraw, pc.H * Φ)
end
Or = reduce(vcat, Oraw)
resp = [norm(Or[:, i]) * sd0[i] for i in 1:n]
println("the innovations' response over the horizon to a one-sd move of each coordinate alone (sd of innovations):")
println("  latents: ", join([@sprintf("%s %.3g", names[i], resp[i]) for i in 1:lay.n_z], ", "))
for (g, rg) in groups
    r = [resp[xr[i]] for i in rg]
    @printf("  %-8s max %.3g (node %d of %d), median %.3g
", g, maximum(r), argmax(r), length(rg), sort(r)[div(end + 1, 2)])
end
println("  discrepancies: ", join([@sprintf("ch%d %.3g", c, resp[dr[c]]) for c in 1:lay.n_ch], ", "))
weak = Dict("latents" => Dict(names[i] => resp[i] for i in 1:lay.n_z),
            "groups" => Dict(g => Dict("max" => maximum(resp[xr[i]] for i in rg), "median" => sort([resp[xr[i]] for i in rg])[div(end + 1, 2)]) for (g, rg) in groups),
            "discrepancies" => [resp[dr[c]] for c in 1:lay.n_ch])
# the core temperature against the core discrepancy: the pair's observability alone
pair = [core_node, dr[2]]
# the pair test: the innovations' response to a one-sd move of the core's field node alone and of the core's
# discrepancy alone, over the horizon; the angle between the two responses says how far the dynamics separate them
Φ = Matrix{Float64}(I, lay.n, lay.n)
resp = Matrix{Float64}[]
for pc in pieces
    global Φ = pc.F * Φ
    size(pc.H, 1) == 0 && continue
    push!(resp, pc.H * Φ[:, pair] .* sd0[pair]')
end
Rp = reduce(vcat, resp)
Sp = svd(Rp).S
θ = acos(min(1.0, abs(dot(Rp[:, 1], Rp[:, 2])) / (norm(Rp[:, 1]) * norm(Rp[:, 2])))) * 180 / π
@printf("the core's field node against its discrepancy over %.1f h: responses of %.3g and %.3g sd, the pair's singular values %.3g and %.3g, the angle between the two responses %.1f degrees\n",
        hours, norm(Rp[:, 1]), norm(Rp[:, 2]), Sp[1], Sp[2], θ)
pairp = [pit_index(lay), dr[1]]
resp = Matrix{Float64}[]
Φ = Matrix{Float64}(I, lay.n, lay.n)
for pc in pieces
    global Φ = pc.F * Φ
    size(pc.H, 1) == 0 && continue
    push!(resp, pc.H * Φ[:, pairp] .* sd0[pairp]')
end
Rq = reduce(vcat, resp)
Sq = svd(Rq).S
θq = acos(min(1.0, abs(dot(Rq[:, 1], Rq[:, 2])) / (norm(Rq[:, 1]) * norm(Rq[:, 2])))) * 180 / π
@printf("the pit against its discrepancy: responses of %.3g and %.3g sd, singular values %.3g and %.3g, angle %.1f degrees\n", norm(Rq[:, 1]), norm(Rq[:, 2]), Sq[1], Sq[2], θq)
open(out, "w") do io
    JSON.print(io, Dict("hours" => hours, "n_updates" => length(recs), "n" => n, "singular_values" => S, "gaps" => gaps,
                        "rank" => Dict(string(thr) => count(>=(thr), S) for thr in (1.0, 0.1, 0.01, 1e-3)), "weakest" => weak,
                        "core_pair" => Dict("responses" => [norm(Rp[:, 1]), norm(Rp[:, 2])], "singular" => Sp, "angle_deg" => θ),
                        "pit_pair" => Dict("responses" => [norm(Rq[:, 1]), norm(Rq[:, 2])], "singular" => Sq, "angle_deg" => θq)))
end
println("wrote ", out)
