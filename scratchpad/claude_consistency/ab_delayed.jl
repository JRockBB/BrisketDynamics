# The online square-root fixed-lag backend (online.jl) on the current 1-D physics through the system adapter,
# on the real-cook mock's first arrivals:
#   the belief layout of a one-slab system is the filter's FilterLayout, entry for entry
#   the dynamics factor's propagation is the monolith's hand-off; the semantic observables are the channels'
#   updates on arrival: convergence, physical beliefs, the innovations, the lag respected
#   fixed lag against the full history on the same readings; a delayed reading against the in-order run
#   sensor subsets, an observable the system cannot supply, a censored reading, a reading outside the lag
#   the restart contract: the snapshot plans as the filter's state does
#   parity with the RxInfer filter at the tangent (n_sigma = 0) over the first arrivals
#   julia --project=estimator estimator/test/test_online.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "..", "estimator", "src", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics
using JSON, LinearAlgebra, Random, Test, ForwardDiff, Printf
const E = BrisketEstimator

REPO = normpath(joinpath(@__DIR__, "..", ".."))
m = JSON.parsefile(joinpath(REPO, "data", "mock", "mock_realcook.json"))
t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]; dt = t[2] - t[1]
pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]), Tpit0=Float64(meta["tpit0_K"]), m0=Float64(meta["m0_kg"]))
cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
blk_old = latent_block(joinpath(REPO, "data", "calibration", "prior_next_cook.json"), cfg)
ev = m["events"]
events_nt = (t_wrap=Float64(ev["t_wrap"]), t_pull=Float64(ev["t_pull"]), t_add=Float64(ev["t_add"]), t_spritz=Float64(ev["t_spritz"]), dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
events = [CookEvent(:wrap, :brisket, ev["t_wrap"]), CookEvent(:pull, :brisket, ev["t_pull"]), CookEvent(:add, :pit, ev["t_add"]),
          CookEvent(:spritz, :brisket, ev["t_spritz"]; dt=ev["dt_spritz"], m=ev["m_spritz"])]
np = realcook_noise_priors(joinpath(REPO, "data", "calibration", "noise_prior_real.json"), dt)
tau(npr) = -dt / log(clamp(npr.ρ_mean, 1e-6, 0.999))
nc = RealCookNoise()
sys = CookSystem(Slab1D(:brisket; cfg); pit)
blk = system_latents(sys, blk_old.prior_phys, blk_old.phys_names; probes=[:brisket])
sensors = [SensorSpec(:pit, PitTemperature(), NoiseModel(white_sd=0.1, ar_sd=np[1].σ_mean, tau=tau(np[1]))),
           SensorSpec(:core, TemperatureAt(:brisket; depth_latent=:core_depth_fraction), NoiseModel(white_sd=0.1, ar_sd=np[2].σ_mean, tau=tau(np[2]))),
           SensorSpec(:humidity, ChamberVapor(), NoiseModel(white_sd=10.0, ar_sd=np[3].σ_mean, tau=tau(np[3]), cap=nc.hum_cap)),
           SensorSpec(:mass, TotalMass(), NoiseModel(white_sd=0.005, ar_sd=np[4].σ_mean, tau=tau(np[4])))]
rd = m["readings"]; mk = m["masks"]
readings = (pit=Float64.(rd["pit"]), core=Float64.(rd["core"]), humidity=Float64.(rd["humidity"]), mass=Float64.(rd["mass"]))
masks = (pit=Bool.(mk["pit"]), core=Bool.(mk["core"]), humidity=Bool.(mk["humidity"]), mass=Bool.(mk["mass"]))
"The readings of the mock up to grid point `kmax` as records, in arrival order."
function records_to(kmax; ids=(:pit, :core, :humidity, :mass))
    recs = MeasurementRecord[]
    for k in 1:kmax, id in ids
        getfield(masks, id)[k] && push!(recs, MeasurementRecord(id, t[k], getfield(readings, id)[k]))
    end
    return recs
end
"The arrivals grouped by time."
arrivals(recs) = [[r for r in recs if r.t == tk] for tk in unique([r.t for r in recs])]
K_END = 8                                                      # the first eight grid points: fourteen minutes
# TEST_ONLINE_ONLY="parity,subsets" runs those testsets alone (their short names below); empty runs every one
const ONLY = filter(!isempty, split(get(ENV, "TEST_ONLINE_ONLY", ""), ","))
run_set(name) = isempty(ONLY) || name in ONLY
recs = records_to(K_END)
clog = CookLog(t, u; events, sensors, measurements=recs)
model = SystemModel(sys, clog, blk; sn=StateNoise())
z0 = [z_moments(p)[1] for p in blk.sites]
x0 = from_z(z0[1:N_PHYS], blk.sites[1:N_PHYS])
AREA = latent_index(blk, "A_meat"); DEPTH = latent_index(blk, "core_depth_fraction")

function run_online(recs; lag, kw...)
    sm = online_init(SystemModel(sys, CookLog(t, u; events, sensors, measurements=recs), blk; sn=StateNoise()); lag, kw...)
    out = NamedTuple[]
    for grp in arrivals(recs)
        push!(out, assimilate!(sm, grp))
    end
    return sm, out
end


ENV["TEST_ONLINE_ONLY"]="none"
late = MeasurementRecord(:pit, t[6], readings.pit[6])
recs_wo = [r for r in recs if !(r.sensor === :pit && r.t == t[6])]
for e in (0, 4)
    E.ONLINE_EXPAND[] = e
    t0 = time()
    sm_full, out_full = run_online(recs; lag=typemax(Int), tol=1e-6, ftol=1e-6)
    rf = out_full[end]
    sm_d, _ = run_online(recs_wo; lag=typemax(Int), tol=1e-6, ftol=1e-6)
    rd_ = assimilate!(sm_d, [late])
    println("expand=$e: converged $(rd_.converged)  |ΔT_pit| $(abs(rd_.T_pit[1] - rf.T_pit[1]))  max |Δz|/sd $(maximum(abs.(rd_.z .- rf.z) ./ sqrt.(diag(rf.z_cov))))  ($(round(time()-t0)) s)")
    flush(stdout)
end
