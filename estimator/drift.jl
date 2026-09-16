# --- the drifting plant: what moves on a real pit, drawn once and walked through the cook -----------
#
# The batch estimator's premise is a constant plant. On a real pit the leak drifts as the vents are nudged,
# the surface coefficients change as the bark sets, and a new charge at the add burns at its own rate with
# its own enthalpy. The drift here is the plant the smoother is judged on (coverage_drift.jl) and the plant
# the pilot can be flown on (mock_closed_loop_realcook.jl): random walks in the unconstrained coordinate
# stepping on the hour (an sd per hour per site), and a jump of the add's charge at the add itself. The
# draws are made once per seed, so the same plant answers every schedule the pilot executes on it; the plant
# is solved in segments (segments.jl) with a boundary on every hour and at the add.

const DRIFT_SD_PER_HOUR = Dict("u_leak" => 0.05, "k_e" => 0.05, "h_c" => 0.03)   # random walks in z, sd per hour
const JUMP_LOG_SD = Dict("r_max" => 0.15, "dH_eff" => 0.10)                     # the add's charge: log-sd about the first charge

"""
    drift_rates(file) -> (sd_per_hour, jump_log_sd)

The rates the FILTER is told: the prior file's `drift` field when earlier cooks have identified them
(`next_cook.py` writes it from the beliefs' walks), else the modelling choice above. The plant keeps its own.
"""
function drift_rates(file::AbstractDict)
    haskey(file, "drift") || return (sd_per_hour=DRIFT_SD_PER_HOUR, jump_log_sd=JUMP_LOG_SD)
    d = file["drift"]
    return (sd_per_hour=Dict(String(k) => Float64(v) for (k, v) in d["sd_per_hour"]),
            jump_log_sd=Dict(String(k) => Float64(v) for (k, v) in get(d, "jump_log_sd", JUMP_LOG_SD)))
end

"""
    DriftDraw(rng, idx; n_hours=30)

The plant's draws: for every drifting site its increment at each hour boundary, for every jump site its jump
at the add, all in z; `idx` maps a site's name to its index in the latent block.
"""
struct DriftDraw
    inc::Dict{Int,Vector{Float64}}
    jump::Dict{Int,Float64}
end
function DriftDraw(rng::AbstractRNG, idx::AbstractDict; n_hours::Int=30)
    inc = Dict{Int,Vector{Float64}}(idx[nm] => DRIFT_SD_PER_HOUR[nm] .* randn(rng, n_hours) for nm in sort(collect(keys(DRIFT_SD_PER_HOUR))))
    jump = Dict{Int,Float64}(idx[nm] => JUMP_LOG_SD[nm] * randn(rng) for nm in sort(collect(keys(JUMP_LOG_SD))))
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
    smoother_moving(idx, t_add) -> moving

The smoother's moving-site spec for this plant: the drift sites with the plant's own rates, and once the add
is set (`t_add` finite and before `ADD_NEVER`) the jump sites at the add with the plant's jump variance.
"""
function smoother_moving(idx::AbstractDict, t_add)
    drift = Any[(idx[nm], :drift, DRIFT_SD_PER_HOUR[nm]^2) for nm in sort(collect(keys(DRIFT_SD_PER_HOUR)))]
    if isfinite(t_add) && t_add < BrisketDynamics.ADD_NEVER
        append!(drift, [(idx[nm], :jump, (float(t_add), JUMP_LOG_SD[nm]^2)) for nm in sort(collect(keys(JUMP_LOG_SD)))])
    end
    return drift
end
