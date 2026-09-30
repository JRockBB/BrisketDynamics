# --- the drifting plant: what moves on a real pit, drawn once and walked through the cook -----------
#
# The batch estimator's premise is a constant plant. On a real pit the leak drifts as the vents are nudged,
# the surface coefficients change as the bark sets, and a new charge at the add burns at its own rate with
# its own enthalpy. The drift here is the plant the smoother is judged on (coverage_drift.jl) and the plant
# the pilot can be flown on (mock_closed_loop_realcook.jl): random walks in the unconstrained coordinate
# stepping on the hour (an sd per hour per site), and a jump of the add's charge at the add itself. The
# draws are made once per seed, so the same plant answers every schedule the pilot executes on it; the plant
# is solved in segments (segments.jl) with a boundary on every hour and at the add.

const DRIFT_SD_PER_HOUR = Dict("u_leak" => 0.05, "k_e" => 0.05, "h_c" => 0.03)   # random walks, sd per hour, stated in the logit of a fraction or the log
const JUMP_LOG_SD = Dict("r_max" => 0.15, "dH_eff" => 0.10)                     # the add's charge: the same, about the first charge

"A rates dict by site name, carried into each site's own coordinate by `drift_scale` when `sites` are given (the constants above are stated in the logit of a bounded site's fraction, the log of a positive one)."
_in_coord(rates, sites, idx) = Dict{String,Float64}(String(k) => Float64(v) * (sites === nothing ? 1.0 : drift_scale(sites[idx[String(k)]])) for (k, v) in rates)

"""
    drift_rates(file; sites=nothing, idx=nothing) -> (sd_per_hour, jump_log_sd)

The rates the FILTER is told: the prior file's `drift` field when earlier cooks have identified them
(`next_cook.py` writes it from the beliefs' walks), else the modelling choice above; in each site's own
coordinate when `sites` and `idx` are given (a site carried in its normal score walks by the same amount in
x). The plant keeps its own (`DriftDraw`, the same scale).
"""
function drift_rates(file::AbstractDict; sites=nothing, idx=nothing)
    d = get(file, "drift", nothing)
    sd = d === nothing ? DRIFT_SD_PER_HOUR : d["sd_per_hour"]
    jp = d === nothing ? JUMP_LOG_SD : get(d, "jump_log_sd", JUMP_LOG_SD)
    return (sd_per_hour=_in_coord(sd, sites, idx), jump_log_sd=_in_coord(jp, sites, idx))
end

"""
    DriftDraw(rng, idx; n_hours=30, sites=nothing)

The plant's draws: for every drifting site its increment at each hour boundary, for every jump site its jump
at the add, all in z; `idx` maps a site's name to its index in the latent block. With `sites` the rates are
carried into each site's own coordinate (`drift_scale`); the standard draws are the same either way, so a
seed's plant walks by the same amount in x whichever coordinate carries a bounded site.
"""
struct DriftDraw
    inc::Dict{Int,Vector{Float64}}
    jump::Dict{Int,Float64}
end
function DriftDraw(rng::AbstractRNG, idx::AbstractDict; n_hours::Int=30, sites=nothing)
    sd = _in_coord(DRIFT_SD_PER_HOUR, sites, idx); jp = _in_coord(JUMP_LOG_SD, sites, idx)
    inc = Dict{Int,Vector{Float64}}(idx[nm] => sd[nm] .* randn(rng, n_hours) for nm in sort(collect(keys(sd))))
    jump = Dict{Int,Float64}(idx[nm] => jp[nm] * randn(rng) for nm in sort(collect(keys(jp))))
    return DriftDraw(inc, jump)
end

"""
    truth_at(z0, dd, te, t_add; from=false) -> z

The block in force at `te` [s] on a plant walked from `z0`: the values step on the hour and at the add.
`from=false` gives the block in force just BEFORE `te` (what the readings up to `te` were made under: a
belief reported at `te` is scored against it); `from=true` the block in force FROM `te` on (the parameters
of a segment that starts there).
"""
function truth_at(z0, dd::DriftDraw, te, t_add; from::Bool=false)
    z = copy(z0)
    h = from ? floor(Int, te / 3600 + 1e-9) : max(ceil(Int, te / 3600 - 1e-9) - 1, 0)
    for (i, inc) in dd.inc
        z[i] += sum(inc[1:min(h, length(inc))])
    end
    if from ? te >= t_add - 1e-9 : te > t_add + 1e-9
        for (i, j) in dd.jump
            z[i] += j
        end
    end
    return z
end

"""
    drift_segments(t, z0, dd, t_add) -> (edges, blocks)

The segment edges (grid indices) and the blocks for `solve_segments` on the grid `t`: a boundary on every
hour and at the add (its grid point at or before `t_add`), each segment at the block in force from its start.
"""
function drift_segments(t, z0, dd::DriftDraw, t_add)
    hours = 3600.0 .* (1:floor(Int, (t[end] - 1e-9) / 3600))
    has_add = t_add < t[end]
    times = has_add ? sort(vcat(hours, t_add)) : hours
    edges = unique(Int[findlast(t .<= τ + 1e-9) for τ in times])
    filter!(e -> 1 < e < length(t), edges)
    t_add_grid = has_add ? t[findlast(t .<= t_add + 1e-9)] : t_add
    blocks = [truth_at(z0, dd, t[e], t_add_grid; from=true) for e in vcat(1, edges)]
    return edges, blocks
end

"""
    solve_drifting(z0, dd, sites, u, t; pit, cfg, A_meat, events...) -> o

The drifting plant under the damper `u` on the grid `t`: the block `z0` (the first `N_PHYS` entries of the
latent block, in z under `sites`) walked by `dd`, solved in segments on the hour and at the add with the
state handed off exactly (`solve_segments`).
"""
function solve_drifting(z0, dd::DriftDraw, sites, u, t; pit::PitConfig, cfg::MeatConfig, A_meat, kw...)
    t_add = get(kw, :t_add, BrisketDynamics.ADD_NEVER)
    edges, blocks = drift_segments(t, z0, dd, t_add)
    xs = [from_z(z[1:N_PHYS], sites[1:N_PHYS]) for z in blocks]
    return solve_segments([x[1:N_THETA] for x in xs], [x[(N_THETA + 1):N_PHYS] for x in xs], u, t, edges; pit, cfg, A_meat, kw...)
end

"""
    smoother_moving(idx, t_add; sites=nothing) -> moving

The smoother's moving-site spec for this plant: the drift sites with the plant's own rates, and once the add
is set (`t_add` finite and before `ADD_NEVER`) the jump sites at the add with the plant's jump variance.
"""
function smoother_moving(idx::AbstractDict, t_add; sites=nothing)
    sd = _in_coord(DRIFT_SD_PER_HOUR, sites, idx); jp = _in_coord(JUMP_LOG_SD, sites, idx)
    drift = Any[(idx[nm], :drift, sd[nm]^2) for nm in sort(collect(keys(sd)))]
    if isfinite(t_add) && t_add < BrisketDynamics.ADD_NEVER
        append!(drift, [(idx[nm], :jump, (float(t_add), jp[nm]^2)) for nm in sort(collect(keys(jp)))])
    end
    return drift
end
