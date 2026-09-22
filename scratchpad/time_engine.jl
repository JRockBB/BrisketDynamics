using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "estimator"))
ENV["JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"] = "1"
include(joinpath(@__DIR__, "..", "estimator", "BrisketEstimator.jl"))
using .BrisketEstimator
using BrisketDynamics, RxInfer, JSON, LinearAlgebra, Printf
const E = BrisketEstimator
REPO = normpath(joinpath(@__DIR__, ".."))
gate_path = joinpath(REPO, "data", "calibration", "estimator_gate_real.json")
d = load_gate(gate_path); raw = JSON.parsefile(gate_path); nz = raw["noise"]
chs = [PitProbe(nz["sigma_pit_eff"], nz["rho_pit"], Bool.(raw["mask_pit"]); data=Float64.(raw["obs_pit"])),
       MeatProbe(nz["sigma_core_eff"], nz["rho_core"], Bool.(raw["mask_core"]); s=0.0, data=Float64.(raw["obs_core"]))]
ph = Physics1D(0); c = copy(d.z_map); n = 15
S, J, lay = physics_outputs(c, d, ph); a, B, w = assemble(chs, c, S, J, d.prior, lay)
E.AFF[] = E.Affine(a, B, copy(c)); E.LM_ANCHOR[] = copy(c)
_, g, h = E.prior_terms(c, d.prior); mp = c .- g ./ h; Λp = Matrix(Diagonal(h)); Λ_lm = Matrix(1e-9 * I, n, n)
# the outer loop's graph solve
graph_posterior_affine(c, a, B, w, d.prior; bp_iters=20)
t0 = time(); for _ in 1:3; graph_posterior_affine(c, a, B, w, d.prior; bp_iters=20); end; @printf("outer's graph solve, 20 message iterations, fresh model: %.2f s\n", (time() - t0) / 3)
t0 = time(); for _ in 1:3; graph_posterior_affine(c, a, B, w, d.prior; bp_iters=1); end; @printf("outer's graph solve, 1 message iteration, fresh model:   %.2f s\n", (time() - t0) / 3)
# the graph engine's infer with a no-op callback, no physics
count = Ref(0)
cb = (on_marginal_update = (ev) -> (count[] += 1; nothing), after_iteration = (ev) -> nothing)
run_it(its) = infer(model = E.cook_graph_lm(), data = (w = w, mp = mp, Λp = Λp, lm_obs = zeros(n), Λ_lm = Λ_lm),
                    meta = E.GRAPH_META_LM, initialization = E.init_z(mp, Λp), iterations = its, callbacks = cb, returnvars = (z = KeepLast(),))
run_it(1)
for its in (1, 5, 20)
    count[] = 0; t0 = time(); run_it(its); @printf("graph engine's infer, %2d iterations (LM node, data prior, callbacks): %.2f s, %d marginal-update callbacks\n", its, time() - t0, count[])
end
cb2 = (after_iteration = (ev) -> nothing,)
run_it2(its) = infer(model = E.cook_graph_lm(), data = (w = w, mp = mp, Λp = Λp, lm_obs = zeros(n), Λ_lm = Λ_lm),
                     meta = E.GRAPH_META_LM, initialization = E.init_z(mp, Λp), iterations = its, callbacks = cb2, returnvars = (z = KeepLast(),))
run_it2(1); t0 = time(); run_it2(20); @printf("same, 20 iterations, no marginal-update callback: %.2f s\n", time() - t0)
t0 = time(); physics_outputs(c, d, ph); @printf("one physics Jacobian: %.2f s\n", time() - t0)
t0 = time(); physics_stack(c, d, ph); @printf("one physics solve: %.2f s\n", time() - t0)
