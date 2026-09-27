# A reduced-order artifact for the bench (docs/estimator-consistency-roadmap.md, section 4): the full order a
# half-thickness brick with its top and sides exposed (`Brick3D`) at the prior's central thickness under a
# brisket-flat footprint, the reduced order the slab of that thickness and top area; the design draws of every
# latent from the pilot's prior (the prior centre added so the envelope brackets it), each solved in two control
# and event regimes over the real-cook mock's cook (its damper and events; the damper perturbed and the wrap moved
# earlier for the second); the error model `ROM − FOM` bias and spread per observable on a ten-minute grid; then
# held-out draws in the training regimes and in a third regime never trained, scored raw against corrected. The
# plan's gate: the corrected held-out RMSE beats the raw ROM error on the core and the mass. The artifact is
# written only when asked (`write 1`), to data/calibration/rom_<name>.json; the numbers are printed either way.
#
# Run:  julia --project=estimator estimator/scripts/train_rom_bench.jl [prior.json] [n_design 24] [n_heldout 8] [hours 14] [seed 1] [nx_slab 20] [write 0/1] [name halfbrick_flat]
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
prior_file = length(ARGS) >= 1 ? ARGS[1] : "prior_next_cook.json"
n_design = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 24
n_heldout = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
hours = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 14.0
seed = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
NX = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 20
do_write = length(ARGS) >= 7 ? ARGS[7] == "1" : false
name = length(ARGS) >= 8 ? ARGS[8] : "halfbrick_flat"

m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t_all = Float64.(m["t"]); u_all = Float64.(m["u"]); meta = m["meta"]
k_end = searchsortedlast(t_all, hours * 3600 + 1e-9)
t = t_all[1:k_end]; u = u_all[1:k_end]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
prior_json = JSON.parsefile(joinpath(REPO, "data", "calibration", prior_file))
prior_phys = read_sites(prior_json["prior"]); phys_names = String.(prior_json["names"])
ev = m["events"]
# the geometry: the prior's central half-thickness, a flat's footprint; no shrinkage on either side (the grid piece's contract)
L_fom = from_z(z_moments(prior_phys[4])[1], prior_phys[4])
Ly, Lz = 0.30, 0.40
cfg = MeatConfig(; nx=NX, n_shrink=0.0, T0=Float64(meta["tcore0_K"]), A_meat=Ly * Lz)
fom = Brick3D(:b; Lx=L_fom, Ly, Lz, nx=6, ny=4, nz=5, exposed=(false, true, true, true, true, true), cfg)
rom_piece = Slab1D(:b; cfg)
sys_fom = CookSystem(fom; pit); sys_rom = CookSystem(rom_piece; pit)
prior_rom = copy(prior_phys)
prior_rom[4] = PriorSite("lognormal", log(L_fom), prior_phys[4].p2)               # the slab's thickness prior centred on the geometry it stands for
blk = system_latents(sys_rom, prior_rom, phys_names; area=false)
z0 = [z_moments(p)[1] for p in blk.sites]
rng = MersenneTwister(5000 + seed)
design = vcat([copy(z0)], [draw_truth(rng, blk.sites) for _ in 1:n_design])
heldout = [draw_truth(rng, blk.sites) for _ in 1:n_heldout]
# the regimes: the mock's cook; the damper perturbed with the wrap an hour earlier; held out, another perturbation with the wrap an hour later and the spritz train denser
t_wrap, t_add, t_spritz, dt_spritz, m_spritz = Float64(ev["t_wrap"]), Float64(ev["t_add"]), Float64(ev["t_spritz"]), Float64(ev["dt_spritz"]), Float64(ev["m_spritz"])
events_mock = [CookEvent(:wrap, :b, t_wrap), CookEvent(:add, :pit, t_add), CookEvent(:spritz, :b, t_spritz; dt=dt_spritz, m=m_spritz)]
u2 = clamp.(u .+ 0.05 .* sin.(range(0, 2π; length=length(t))), 0.02, 0.98)
events2 = [CookEvent(:wrap, :b, t_wrap - 3600.0), CookEvent(:add, :pit, t_add), CookEvent(:spritz, :b, t_spritz; dt=dt_spritz, m=m_spritz)]
u3 = clamp.(u .- 0.04 .* cos.(range(0, 1.5π; length=length(t))), 0.02, 0.98)
events3 = [CookEvent(:wrap, :b, t_wrap + 3600.0), CookEvent(:add, :pit, t_add - 1800.0), CookEvent(:spritz, :b, t_spritz - 1800.0; dt=0.75 * dt_spritz, m=m_spritz)]
train_scenarios = [(u_knots=u, u_tknots=t, tsave=t, events=events_mock), (u_knots=u2, u_tknots=t, tsave=t, events=events2)]
heldout_scenarios = [(u_knots=u3, u_tknots=t, tsave=t, events=events3)]
t_grid = collect(0.0:600.0:t[end])
println("the bench's reduced-order artifact: FOM a half-thickness brick Lx ", round(L_fom; digits=4), " m × ", Ly, " × ", Lz, " (", fom.mesh.N, " nodes, top and sides exposed); ROM the slab at nx ", NX,
        "; ", length(design), " design draws (the prior centre first) × 2 regimes, ", n_heldout, " held-out draws × 3 regimes; ", hours, " h on the mock's damper; seed ", seed)
flush(stdout)
t0 = time()
r = train_rom(name, sys_fom, sys_rom, blk, design, u, t, t; t_grid, events=events_mock, scenarios=train_scenarios, state_noise=StateNoise())
@printf("trained in %.0f s; at %.0f h the core's bias %.2f K ± %.2f, the surface's %.2f ± %.2f, the mass's %.1f g ± %.1f, the chamber vapor's %.0f Pa ± %.0f, the heat load's %.1f W ± %.1f\n",
        time() - t0, t_grid[end] / 3600, r.errors[:Tcore].bias[end], r.errors[:Tcore].sd[end], r.errors[:Tsurf].bias[end], r.errors[:Tsurf].sd[end],
        1000 * r.errors[:mass].bias[end], 1000 * r.errors[:mass].sd[end], r.errors[:pv_pit].bias[end], r.errors[:pv_pit].sd[end], r.errors[:q_meat].bias[end], r.errors[:q_meat].sd[end])
flush(stdout)
t1 = time()
v_in = validate_rom(r, sys_fom, sys_rom, blk, heldout, u, t, t; events=events_mock, scenarios=train_scenarios, width=2.0)
v_out = validate_rom(r, sys_fom, sys_rom, blk, heldout, u, t, t; events=events_mock, scenarios=heldout_scenarios, width=2.0)
@printf("validated in %.0f s\n", time() - t1)
units = Dict(:Tcore => "K", :Tsurf => "K", :mass => "kg", :pv_pit => "Pa", :q_meat => "W")
gate = true
for (lab, v) in (("held-out draws, the training regimes", v_in), ("held-out draws, a regime never trained", v_out))
    println(lab, ":")
    @printf("  %-8s %10s %10s %10s %10s  skill\n", "output", "raw rms", "corr rms", "max |e|", "within 2sd")
    for obs in ROM_OBSERVABLES
        x = v[obs]
        skill = x.raw_rms > 0 ? 1 - x.rms / x.raw_rms : NaN
        @printf("  %-8s %10.3g %10.3g %10.3g %10.2f  %+.2f [%s]\n", obs, x.raw_rms, x.rms, x.max, x.coverage, skill, units[obs])
        (obs in (:Tcore, :mass) && !(x.rms < x.raw_rms)) && (global gate = false)
    end
    println("  every held-out draw inside the envelope: ", v[:in_envelope].coverage == 1.0)
end
println(gate ? "the gate holds: the corrected held-out error beats the raw reduction error on the core and the mass in both regimes" :
               "the gate FAILS: the correction does not beat the raw reduction error on the core or the mass in some regime")
if do_write
    path = joinpath(REPO, "data", "calibration", "rom_$(name).json")
    save_rom(path, r)
    println("wrote ", path, " (fingerprint ", r.fingerprint, ")")
else
    println("not written (write 1 to write data/calibration/rom_$(name).json)")
end
