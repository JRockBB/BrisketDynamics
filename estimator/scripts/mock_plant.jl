# Mock plants for the streaming estimator: the readings a cook would
# produce, on the real cook's damper log and ambient, from
#   (a) the monolith at "true" latents the estimator can represent, and
#   (b) the 2-D engine — a structurally different plant (finite cylinder,
#       true cold spot, 2-D shrink) — with the same noise,
# so the estimator can be watched succeeding where the model is right and
# failing where it is not. The true cut is thicker than the calibrated
# one (2x), draws less heat (h_c x0.6), has a larger exposed area (x1.5),
# and the core probe sits off-center (0.8 L from the top).
#
# Run:  julia --project=estimator estimator/scripts/mock_plant.jl
# Writes data/mock/mock_1d.json and data/mock/mock_2d.json.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf

REPO = normpath(joinpath(@__DIR__, "..", ".."))
gate_path = joinpath(REPO, "data", "calibration", "estimator_gate_real.json")
d = load_gate(gate_path)
raw = JSON.parsefile(gate_path)
nz = raw["noise"]
t, u = d.t, d.u
nt = length(t)
x_map = from_z(d.z_map, d.prior)

# --- the true cut
θ_true = copy(x_map[1:8]); θ_true[3] *= 0.6; θ_true[4] *= 2.0
φ_true = x_map[9:15]
L_true = θ_true[4]
A_true = 1.5 * d.cfg.A_meat
depth_fraction_true = 0.8
depth_true = probe_depth(L_true, depth_fraction_true)      # from the top face
RHO_D = 320.0
W0, F0 = d.cfg.W0, d.cfg.F0
noise = Dict("pit" => (nz["sigma_pit_eff"], nz["rho_pit"]),
             "core" => (nz["sigma_core_eff"], nz["rho_core"]),
             "humidity" => (100.0, 0.8), "ir" => (2.0, 0.8), "mass" => (0.02, 0.5))
# `--altnoise`: the same plants with a different true noise (a noisier pit probe, a quieter
# and less correlated core probe, a worse humidity sensor, a better scale), for the noise
# identification runs; files get the suffix _alt
ALT = "--altnoise" in ARGS
if ALT
    noise = Dict("pit" => (3.0, 0.6), "core" => (1.0, 0.85),
                 "humidity" => (200.0, 0.5), "ir" => (2.0, 0.8), "mass" => (0.01, 0.2))
end
SUFFIX = ALT ? "_alt" : ""

function ar1_noise(rng, n, σ, ρ)
    e = zeros(n)
    e[1] = σ * randn(rng)
    for k in 2:n
        e[k] = ρ * e[k - 1] + σ * sqrt(1 - ρ^2) * randn(rng)
    end
    return e
end

function noisy(rng, clean::Dict)
    out = Dict{String,Vector{Float64}}()
    for (k, v) in clean
        σ, ρ = noise[k]
        out[k] = v .+ ar1_noise(rng, length(v), σ, ρ)
    end
    return out
end

interp1(v, pos) = (i = clamp(floor(Int, pos), 1, length(v) - 1); λ = pos - i; (1 - λ) * v[i] + λ * v[i + 1])

truth = Dict("names" => vcat(d.names, ["A_meat", "core_depth_fraction"]),
             "x" => vcat(θ_true, φ_true, [A_true, depth_fraction_true]))
base = Dict("t" => t, "u" => u, "meta" => raw["meta"], "truth" => truth,
            "noise" => Dict(k => Dict("sigma" => v[1], "rho" => v[2]) for (k, v) in noise))

# --- (a) the monolith at the true latents
t0 = time()
o = monolith_solve(θ_true, φ_true, u, t, t; pit=d.pit, cfg=d.cfg, A_meat=A_true)
nx = size(o.T_field, 1)
w = BrisketEstimator.fv_weights(nx)
s_probe = 1 - depth_true / L_true
pos = 1 + s_probe * (nx - 1)
clean1 = Dict(
    "pit" => collect(o.Tpit),
    "core" => [interp1(o.T_field[:, k], pos) for k in 1:nt],
    "humidity" => collect(o.pv_pit),
    "ir" => collect(o.Tsurf),
    "mass" => [meat_mass_1d(dot(w, o.W_field[:, k]), dot(w, o.F_field[:, k]), A_true, L_true, d.cfg) for k in 1:nt],
)
rng = MersenneTwister(11)
mock1 = merge(base, Dict("plant" => "monolith (1-D)", "clean" => clean1, "readings" => noisy(rng, clean1)))
@printf("1-D plant: %.1f s; core at s = %.2f; mass %.2f -> %.2f kg\n", time() - t0, s_probe,
        clean1["mass"][1], clean1["mass"][end])

# --- (b) the 2-D engine: a finite cylinder of height 2 L_true whose exposed
#         area equals A_true (2πR² + 2πR·Hz = A_true), probe on the axis
Hz = 2 * L_true
R = (-Hz + sqrt(Hz^2 + 2 * A_true / π)) / 2
θ2 = θ_true[[1, 2, 3, 5, 6, 7, 8]]
nr, nzc = 16, 14
t0 = time()
o2 = coupled_solve_2d(θ2, φ_true, u, t, t; R=R, Hz=Hz, nr=nr, nz=nzc, cfg=d.cfg, pit=d.pit,
                      hum=BrisketDynamics.Humidity(), h=30.0, n_meat=1.0)
Δr, Δz = R / (nr - 1), Hz / (nzc - 1)
rV = [i == 1 ? (Δr / 2)^2 / 2 : i == nr ? (R^2 - (R - Δr / 2)^2) / 2 : (i - 1) * Δr * Δr for i in 1:nr]
zw = [j == 1 || j == nzc ? 0.5 : 1.0 for j in 1:nzc]
vol = [2π * rV[i] * zw[j] * Δz for i in 1:nr, j in 1:nzc]
zp = Hz - depth_true
posz = 1 + zp / Δz
clean2 = Dict(
    "pit" => collect(o2.Tpit),
    "core" => [interp1(o2.T_field[1, :, k], posz) for k in 1:nt],
    "humidity" => collect(o2.pv_pit),
    "ir" => [o2.T_field[1, nzc, k] for k in 1:nt],
    "mass" => [RHO_D * sum(vol .* meat_mass_factor.(o2.W_field[:, :, k], o2.F_field[:, :, k], Ref(d.cfg))) for k in 1:nt],
)
rng = MersenneTwister(12)
mock2 = merge(base, Dict("plant" => "2-D axisymmetric (structural mismatch)", "clean" => clean2,
                         "readings" => noisy(rng, clean2),
                         "geometry" => Dict("R" => R, "Hz" => Hz, "nr" => nr, "nz" => nzc)))
@printf("2-D plant: %.1f s; R = %.3f m, Hz = %.3f m; mass %.2f -> %.2f kg; final core %.1f / axis-mid %.1f C\n",
        time() - t0, R, Hz, clean2["mass"][1], clean2["mass"][end], clean2["core"][end] - 273.15,
        o2.Tcore[end] - 273.15)

dir = joinpath(REPO, "data", "mock")
mkpath(dir)
for (name, m) in (("mock_1d" * SUFFIX * ".json", mock1), ("mock_2d" * SUFFIX * ".json", mock2))
    open(joinpath(dir, name), "w") do io
        JSON.print(io, m)
    end
    println("wrote ", joinpath(dir, name))
end
