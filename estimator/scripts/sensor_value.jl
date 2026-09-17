# The sensor-value table: what each candidate channel buys, on the
# calibrated real cook, before anyone buys the instrument. Fisher
# information from the channel Jacobians at the MAP — no data needed —
# with the core probe's depth as an extra latent (people rarely hit the
# exact center), so the thickness / probe-depth ridge is on the table.
#
# Run:  julia --project=estimator estimator/scripts/sensor_value.jl
# Writes data/calibration/sensor_value_real.json.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using JSON, LinearAlgebra, Printf

REPO = normpath(joinpath(@__DIR__, "..", ".."))
gate_path = joinpath(REPO, "data", "calibration", "estimator_gate_real.json")
d = load_gate(gate_path)
raw = JSON.parsefile(gate_path)
mask_pit = Bool.(raw["mask_pit"])
mask_core = Bool.(raw["mask_core"])
nz = raw["noise"]
σp, ρp, σc, ρc = nz["sigma_pit_eff"], nz["rho_pit"], nz["sigma_core_eff"], nz["rho_core"]
nt = length(d.t)
L_map = exp(d.z_map[4])

# latents: the 15 physics sites + the exposed area (prior median the
# config's 0.12 m², 30 %) + the core probe's depth from the surface
# (prior median 0.8 L, 30 % — "about the middle")
A_cfg = d.cfg.A_meat
extra = [LatentSpec("A_meat", "lognormal", log(A_cfg), 0.3),
         LatentSpec("core_depth", "lognormal", log(0.8 * L_map), 0.3)]
prior = vcat(d.prior, [e.site for e in extra])
names = vcat(d.names, [e.name for e in extra])
z = vcat(d.z_map, [log(A_cfg), log(0.8 * L_map)])
AREA = length(z) - 1
DEPTH = length(z)

t0 = time()
S, J, lay = physics_outputs(z, d; area_latent=AREA)
@printf("stacked outputs: %d rows, Jacobian %d x %d (%.1f s incl. compile)\n", length(S), size(J)..., time() - t0)

# --- consistency with the gate: pit + core-at-center must give the gate's rows
pit = PitProbe(σp, ρp, mask_pit)
core_center = MeatProbe(σc, ρc, mask_core; s=0.0)
_, Bc, _ = assemble([pit, core_center], z, S, J, prior, lay)
_, Jz = physics_mean_jac(d.z_map, d)
Bg = d.L * Jz
@printf("registry vs gate rows: max |dB| / max |B| = %.2e  (%d x %d)\n",
        maximum(abs.(Bc[:, 1:N_PHYS] .- Bg)) / maximum(abs.(Bg)), size(Bg)...)
# conditioned on the area and the depth (the gate had the area fixed at the
# config's value and the probe at the center)
covc, sdc = fisher_cov([pit, core_center], z, S, J, prior, lay; fix=[AREA, DEPTH])
sd_gate = sqrt.(diag(d.cov_map))
@printf("registry vs gate Fisher sd (area, depth fixed), max rel diff over the 15 sites: %.2e\n",
        maximum(abs.(sdc[1:N_PHYS] .- sd_gate) ./ sd_gate))

# --- the candidate channels
full = trues(nt)
ends = falses(nt); ends[1] = true; ends[end] = true
core_unknown = MeatProbe(σc, ρc, mask_core; depth_latent=DEPTH)
second_probe = MeatProbe(σc, ρc, full; depth=0.010)            # 1 cm below the surface, depth known
humidity = HumiditySensor(100.0, 0.8, full)                          # ~5 % RH, 10-min grid
surface_ir = SurfaceIR(2.0, 0.8, full)
scale_cont = MassScale(0.02, 0.5, full; area_latent=AREA)      # 20 g, every 10 min
scale_ends = MassScale(0.02, 0.0, ends; area_latent=AREA)      # before / after only
ruler = StaticReading(4, 0.003)                                # 3 mm on the half-thickness
footprint = StaticReading(AREA, 0.01)                          # 0.01 m² (~8 %) on the exposed area
depth_ruler = StaticReading(DEPTH, 0.003)                      # the insertion depth measured
fuel = FuelScale(0.05, 0.5, full)                              # a load cell under the firebox, 50 g
hourly = falses(nt); hourly[1:6:end] .= true
bark = BarkLightness(3.0, 0.8, hourly)                         # a colorimeter on the bark, 3 L*, hourly
wetbulb = WetBulb(1.0, 0.8, full)                              # a wetted thermocouple, 1 K

sets = [
    ("pit + core at center (as calibrated)", [pit, core_center]),
    ("pit + core, depth unknown", [pit, core_unknown]),
    ("+ ruler on thickness", [pit, core_unknown, ruler]),
    ("+ insertion depth measured", [pit, core_unknown, depth_ruler]),
    ("+ second probe, 1 cm, known depth", [pit, core_unknown, second_probe]),
    ("+ humidity sensor", [pit, core_unknown, humidity]),
    ("+ surface IR", [pit, core_unknown, surface_ir]),
    ("+ fuel scale", [pit, core_unknown, fuel]),
    ("+ bark colorimeter, hourly", [pit, core_unknown, bark]),
    ("+ wet-bulb probe", [pit, core_unknown, wetbulb]),
    ("+ scale, before/after", [pit, core_unknown, scale_ends]),
    ("+ scale, continuous", [pit, core_unknown, scale_cont]),
    ("+ scale, continuous + footprint ruler", [pit, core_unknown, scale_cont, footprint]),
    ("+ rulers (thickness, footprint) + scale + second probe",
     [pit, core_unknown, ruler, footprint, second_probe, scale_cont]),
    ("+ everything", [pit, core_unknown, ruler, footprint, depth_ruler, second_probe, humidity,
                      surface_ir, scale_cont, fuel, bark, wetbulb]),
]
results = []
covs = Dict{String,Matrix{Float64}}()
for (label, chs) in sets
    cov, sd = fisher_cov(chs, z, S, J, prior, lay)
    covs[label] = cov
    push!(results, (label=label, sd=sd, logdet=logdet(Symmetric(cov))))
end
base = results[2]

println()
@printf("%-12s %9s", "site", "sd base")
for r in results[3:end]
    @printf(" %7s", "")
end
println()
header = ["sd base"; ["set $(i)" for i in 3:length(results)]]
println("sd ratios to 'pit + core, depth unknown' (set 2); columns are the sets in order:")
for (i, r) in enumerate(results)
    @printf("  set %2d  %s\n", i, r.label)
end
println()
@printf("%-12s %9s", "site", "sd(set2)")
for i in 3:length(results)
    @printf(" %7s", "set $i")
end
println()
for j in eachindex(names)
    @printf("%-12s %9.4f", names[j], base.sd[j])
    for r in results[3:end]
        @printf(" %7.3f", r.sd[j] / base.sd[j])
    end
    println()
end
println()
println("information gained vs set 2 [nats, half log-det ratio]:")
for (i, r) in enumerate(results)
    i == 2 && continue
    @printf("  set %2d  %6.2f  %s\n", i, 0.5 * (base.logdet - r.logdet), r.label)
end

out = Dict(
    "names" => names,
    "z_eval" => z,
    "sets" => [Dict("label" => r.label, "sd" => r.sd, "logdet" => r.logdet) for r in results],
    "baseline_set" => 2,
    "noise" => Dict("sigma_pit" => σp, "rho_pit" => ρp, "sigma_core" => σc, "rho_core" => ρc,
                    "humidity_Pa" => 100.0, "ir_K" => 2.0, "scale_kg" => 0.02, "ruler_m" => 0.003),
    "second_probe_depth_m" => 0.010,
    "core_depth_prior" => Dict("median_m" => 0.8 * L_map, "sd_log" => 0.3),
    "area_prior" => Dict("median_m2" => A_cfg, "sd_log" => 0.3, "footprint_ruler_m2" => 0.01),
)
dst = joinpath(REPO, "data", "calibration", "sensor_value_real.json")
open(dst, "w") do io
    JSON.print(io, out, 1)
end
println("wrote ", dst)
