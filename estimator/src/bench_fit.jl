# The bench's identification step for the factor-graph backend, in Julia: the state's process-noise rates and the
# sensors' OU and white variances fitted to the innovations' autocovariances from the accumulators the workers
# wrote (als.jl, fed by online.jl's `ALSFeed`), and the calibration file the fresh round and every driver read.
# The rules are the frozen Python fit's (estimator/python/als_fit.py, the RxInfer bench's oracle), restated:
#
#   * the rows are weighted by their own residual variance per channel pair and kind (the squares at one update,
#     the products across updates), feasibly: the weights start at one, each block's residual variance at the
#     unbounded fit re-weights it, repeated until the weights hold;
#   * a group enters the fit when the innovations see it: its own column pins its variance rate to a relative
#     standard error under `ALS_REL_SE_MAX`, or the unbounded fit moves it by more than `ALS_MOVE_SIGMAS` standard
#     errors; a group the innovations cannot see keeps its numbers and is reported;
#   * the fit is bounded: no variance rate leaves [θ/cap², θ·cap²], none is negative;
#   * after the bounded fit a group whose joint standard error exceeds its fitted variance rate without a
#     significant move is held at its number and the fit is solved again;
#   * the initial widths are not identified (their transient is one; the rates recur) and are kept;
#   * the standard errors are quoted twice: from the model's rows (the products as independent) and from a
#     leave-one-cook-out jackknife over the cooks' own statistics (a cook's products share its plant and belief).
#
# The fit needs no truth, so the same step applies to a real cook's innovations. Included by BrisketEstimator
# after online.jl.

export als_pool, als_fit, als_lag_table, format_als_fit, format_als_lags,
       SYSTEM_STATE_NOISE_FILE, system_calibration_record, write_system_calibration, load_system_calibration

const ALS_WEIGHT_ROUNDS = 6
const ALS_REL_SE_MAX = 0.5
const ALS_MOVE_SIGMAS = 2.0
const ALS_REL_SE_HOLD = 1.0

_als_mat(rows) = isempty(rows) ? zeros(0, 0) : Matrix{Float64}(reduce(hcat, [Float64.(r) for r in rows])')
_als_finite(v) = isfinite(v) ? Float64(v) : nothing

"""
    als_pool(summaries) -> NamedTuple

The cooks' accumulators (`als_summary` dictionaries, or the same parsed from a record) pooled: the same groups,
variance rates, window and bin on every cook; the normal equations, the cost at δθ = 0, the counts, the blocks
per channel pair and kind, the pairs per channel pair and lag bin.
"""
function als_pool(summaries::AbstractVector)
    isempty(summaries) && throw(ArgumentError("no accumulator summaries to pool"))
    first_ = summaries[1]
    groups = String.(first_["groups"]); theta = Float64.(first_["theta"]); n_g = length(groups)
    G = zeros(n_g, n_g); h = zeros(n_g); rss = 0.0; n_rows = 0; n_updates = 0
    pairs = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
    blocks = Dict{Tuple{Int,Int,Bool},ALSBlock}()
    for s in summaries
        (String.(s["groups"]) == groups && isapprox(Float64.(s["theta"]), theta) && s["lag_s"] == first_["lag_s"] && s["bin_s"] == first_["bin_s"]) ||
            throw(ArgumentError("the cooks' accumulators disagree on the groups, the rates, the window or the bin"))
        G .+= _als_mat(s["G"]); h .+= Float64.(s["h"]); rss += Float64(s["rss"])
        n_rows += Int(s["n_rows"]); n_updates += Int(s["n_updates"])
        for p in s["pairs"]
            v = get!(pairs, (Int(p["a"]), Int(p["b"]), Int(p["bin"]))) do; zeros(4 + n_g); end
            v[1] += p["n"]; v[2] += p["sum_product"]; v[3] += p["sum_product2"]; v[4] += p["sum_base"]; v[5:end] .+= Float64.(p["sum_b"])
        end
        for b in s["blocks"]
            blk = get!(blocks, (Int(b["a"]), Int(b["b"]), Bool(b["same"]))) do; ALSBlock(n_g); end
            blk.G .+= _als_mat(b["G"]); blk.h .+= Float64.(b["h"]); blk.rss += Float64(b["rss"])
            blk.n += Int(b["n"]); blk.s_sum += Float64(b["s_sum"]); blk.s_sq += Float64(b["s_sq"])
        end
    end
    return (groups=groups, theta=theta, G=G, h=h, rss=rss, n_rows=n_rows, n_updates=n_updates, n_cooks=length(summaries),
            lag_s=Float64(first_["lag_s"]), bin_s=Float64(first_["bin_s"]), pairs=pairs, blocks=blocks)
end

# the bounded least squares min δᵀGδ − 2hᵀδ over lo ≤ δ ≤ hi by an active set on the normal equations (a few
# dozen variables at most)
function _bounded_ls(G::AbstractMatrix, h::AbstractVector, lo::AbstractVector, hi::AbstractVector; iters::Int=200)
    n = length(h)
    n == 0 && return zeros(0)
    Gs = Matrix(Symmetric(0.5 .* (G .+ G'))) + (1e-12 * max(tr(G) / n, 1e-300)) * I
    x = zeros(n); fixed = falses(n)
    for i in 1:n
        hi[i] - lo[i] <= 1e-15 * max(abs(lo[i]), abs(hi[i]), 1e-300) && (x[i] = lo[i]; fixed[i] = true)
    end
    for _ in 1:iters
        free = findall(.!fixed); fx = findall(fixed)
        if !isempty(free)
            rhs = h[free] .- Gs[free, fx] * x[fx]
            x[free] = Gs[free, free] \ rhs
        end
        viol = [i for i in free if x[i] < lo[i] || x[i] > hi[i]]
        if !isempty(viol)
            for i in viol; x[i] = clamp(x[i], lo[i], hi[i]); fixed[i] = true; end
            continue
        end
        grad = Gs * x .- h                                              # half the gradient of the cost
        release = 0; worst = 0.0
        for i in fx
            v = x[i] <= lo[i] ? -grad[i] : grad[i]                      # at the lower bound the cost must not fall going up; at the upper going down
            v > worst && (release = i; worst = v)
        end
        release == 0 && break
        fixed[release] = false
    end
    return clamp.(x, lo, hi)
end

function _als_weighted(blocks, weights, n_g)
    G = zeros(n_g, n_g); h = zeros(n_g); rss = 0.0
    for (k, b) in blocks
        w = weights[k]
        G .+= w .* b.G; h .+= w .* b.h; rss += w * b.rss
    end
    return G, h, rss
end
function _als_block_variances(blocks, δ)
    out = Dict{Tuple{Int,Int,Bool},Float64}()
    for (k, b) in blocks
        r = b.rss - 2 * dot(b.h, δ) + dot(δ, b.G * δ)
        out[k] = b.n >= 2 ? max(r, 0.0) / max(b.n - 1, 1) : NaN
    end
    return out
end

"""
    als_fit(pooled; cap=4.0, summaries=nothing) -> Dict

The rates from the pooled statistics: per group the variance rate before and after, the factor on the rate, the
standard error of the rate, whether it was identified and whether the cap bound; the fit's cost before and after
and the rows' noise; the marginal relative standard error and the move in standard errors that judged each
group; per block its weight and residual variance at the fit; with the cooks' `summaries` (three or more) the
leave-one-cook-out jackknife of the whole procedure.
"""
function als_fit(pooled; cap::Float64=4.0, summaries=nothing)
    groups = pooled.groups; θ = copy(pooled.theta); n_g = length(groups)
    N = pooled.n_rows
    N >= n_g + 2 || throw(ArgumentError("$N rows cannot identify $n_g rates"))
    blocks = pooled.blocks
    isempty(blocks) && throw(ArgumentError("the pooled statistics carry no blocks"))
    weights = Dict(k => 1.0 for k in keys(blocks))
    G, h, rss0 = _als_weighted(blocks, weights, n_g)
    seen = [g for g in 1:n_g if G[g, g] > 0]
    for _ in 1:ALS_WEIGHT_ROUNDS
        du = zeros(n_g)
        if !isempty(seen)
            Gs = G[seen, seen]; Gs = Gs + (1e-12 * tr(Gs) / length(seen)) * I
            du[seen] = Gs \ h[seen]
        end
        var = _als_block_variances(blocks, du)
        new = Dict(k => (isfinite(v) && v > 0 ? 1 / v : weights[k]) for (k, v) in var)
        changed = maximum(abs(new[k] / weights[k] - 1) for k in keys(blocks))
        weights = new
        G, h, rss0 = _als_weighted(blocks, weights, n_g)
        changed < 1e-3 && break
    end
    d = diag(G)
    seen = [g for g in 1:n_g if d[g] > 0]
    σ2 = rss0 / (N - 1)
    moved = zeros(n_g)
    if !isempty(seen)
        Gs = G[seen, seen]; Gs = Gs + (1e-12 * tr(Gs) / length(seen)) * I
        du = Gs \ h[seen]
        rss_u = rss0 - 2 * dot(h[seen], du) + dot(du, Gs * du)
        σ2 = max(rss_u, 0.0) / max(N - length(seen), 1)
        se_u = sqrt.(max.(diag(inv(Gs)) .* σ2, 0.0))
        moved[seen] = abs.(du) ./ max.(se_u, 1e-300)
    end
    rel_marginal = [d[g] > 0 && θ[g] > 0 ? sqrt(σ2 / d[g]) / θ[g] : Inf for g in 1:n_g]
    active = [g for g in seen if rel_marginal[g] < ALS_REL_SE_MAX || moved[g] > ALS_MOVE_SIGMAS]
    held = Dict{String,String}(groups[g] => "unidentified" for g in 1:n_g if !(g in active))
    δ = zeros(n_g); se_θ = fill(NaN, n_g); rss1 = rss0
    while true
        a = active
        lo = θ[a] ./ cap^2 .- θ[a]; hi = θ[a] .* cap^2 .- θ[a]
        da = isempty(a) ? zeros(0) : _bounded_ls(G[a, a], h[a], lo, hi)
        δ = zeros(n_g); δ[a] = da
        rss1 = rss0 - 2 * dot(h, δ) + dot(δ, G * δ)
        σ2 = rss1 / max(N - length(a), 1)
        se_θ = fill(NaN, n_g)
        if !isempty(a)
            Ga = G[a, a]
            cov = σ2 .* inv(Ga + (1e-12 * tr(Ga) / length(a)) * I)
            se_θ[a] = sqrt.(max.(diag(cov), 0.0))
        end
        worst = 0; wrel = 0.0
        for g in active
            rel = se_θ[g] / max(θ[g] + δ[g], 1e-300)
            if rel > ALS_REL_SE_HOLD && moved[g] < ALS_MOVE_SIGMAS && rel > wrel
                worst = g; wrel = rel
            end
        end
        worst == 0 && break
        active = [g for g in active if g != worst]
        held[groups[worst]] = "held: the fit could not tell its sign"
    end
    θn = θ .+ δ
    rates_new = sqrt.(max.(θn, 0.0)); rates = sqrt.(θ)
    bvar = _als_block_variances(blocks, δ)
    at_cap = [g in active && (abs(θn[g] - θ[g] / cap^2) <= 1e-9 * θ[g] || abs(θn[g] - θ[g] * cap^2) <= 1e-9 * θ[g]) for g in 1:n_g]
    jk = summaries !== nothing && length(summaries) >= 3 ? _als_jackknife(summaries, cap) : nothing
    bkeys = sort(collect(keys(blocks)))
    return Dict{String,Any}(
        "jackknife" => jk, "groups" => groups, "theta" => θ, "theta_new" => θn, "rates" => rates, "rates_new" => rates_new,
        "factors" => [rates[g] > 0 ? rates_new[g] / rates[g] : 1.0 for g in 1:n_g],
        "se_rate" => [rates_new[g] > 0 ? _als_finite(se_θ[g] / (2 * rates_new[g])) : nothing for g in 1:n_g],
        "identified" => [g in active for g in 1:n_g], "held" => held, "cap_bound" => at_cap,
        "rel_se_marginal" => [_als_finite(v) for v in rel_marginal], "moved_sigmas" => [_als_finite(v) for v in moved],
        "rss_before" => rss0, "rss_after" => rss1, "n_rows" => N, "sigma2" => σ2,
        "blocks" => [Dict("a" => k[1], "b" => k[2], "same" => k[3], "n" => blocks[k].n, "weight" => _als_finite(weights[k]), "variance_at_fit" => _als_finite(bvar[k])) for k in bkeys],
        "n_cooks" => pooled.n_cooks, "n_updates" => pooled.n_updates, "lag_s" => pooled.lag_s, "bin_s" => pooled.bin_s, "cap" => cap)
end

function _als_jackknife(summaries::AbstractVector, cap::Float64)
    n = length(summaries)
    fits = [als_fit(als_pool([summaries[j] for j in 1:n if j != i]); cap) for i in 1:n]
    θ = Matrix(reduce(hcat, [Float64.(f["theta_new"]) for f in fits])')        # n × n_g
    r = Matrix(reduce(hcat, [Float64.(f["rates_new"]) for f in fits])')
    se_θ = sqrt.((n - 1) / n .* vec(sum((θ .- sum(θ; dims=1) ./ n) .^ 2; dims=1)))
    se_r = sqrt.((n - 1) / n .* vec(sum((r .- sum(r; dims=1) ./ n) .^ 2; dims=1)))
    n_g = size(θ, 2)
    return Dict{String,Any}("n_cooks" => n, "se_theta" => [_als_finite(v) for v in se_θ], "se_rate" => [_als_finite(v) for v in se_r],
                            "identified_in" => [count(f["identified"][g] for f in fits) for g in 1:n_g],
                            "at_cap_in" => [count(f["cap_bound"][g] for f in fits) for g in 1:n_g],
                            "rates_min" => vec(minimum(r; dims=1)), "rates_max" => vec(maximum(r; dims=1)))
end

"""
    als_lag_table(pooled, fit, channels, bins) -> Vector{Dict}

Per channel (its own products) and lag bin: the sample mean product, the model at the rates the round ran with
and at the fitted rates, the standard error of the sample mean.
"""
function als_lag_table(pooled, fit, channels::AbstractDict{Int,String}, bins::AbstractVector{Int})
    δ = Float64.(fit["theta_new"]) .- Float64.(fit["theta"])
    rows = Dict{String,Any}[]
    for (c, lab) in sort(collect(channels); by=first), bn in bins
        v = get(pooled.pairs, (c, c, bn), nothing)
        (v === nothing || v[1] < 2) && continue
        n = v[1]; sample = v[2] / n; base = v[4] / n; model_new = base + dot(v[5:end], δ) / n
        var_p = max(v[3] / n - sample^2, 0.0) * n / (n - 1)
        se = n >= 3 ? sqrt(var_p / n) : sqrt((bn == 0 ? 2.0 : 1.0) / n)
        push!(rows, Dict("channel" => lab, "bin" => bn, "lag_s" => bn * pooled.bin_s, "n" => Int(n), "sample" => sample, "se" => se,
                         "model_before" => base, "model_after" => model_new))
    end
    return rows
end

"The fit as the driver prints it."
function format_als_fit(fit, units::AbstractDict{String,String}, channels::AbstractDict{Int,String}=Dict{Int,String}())
    lines = String[]
    push!(lines, @sprintf("  the autocovariance fit: %d products over %s cooks (%s updates), lags to %.1f h; weighted cost %.0f before, %.0f after (the weighted rows' noise %.2f, one when the weights are right)",
                          fit["n_rows"], string(fit["n_cooks"]), string(fit["n_updates"]), fit["lag_s"] / 3600, fit["rss_before"], fit["rss_after"], fit["sigma2"]))
    own = [b for b in fit["blocks"] if b["variance_at_fit"] !== nothing && b["a"] == b["b"]]
    push!(lines, "  the blocks' residual variance at the fit (their weight is its inverse): " *
                 join([@sprintf("%s×%s %s %.3g (n %d)", get(channels, b["a"], string(b["a"])), get(channels, b["b"], string(b["b"])), b["same"] ? "at one update" : "across updates", b["variance_at_fit"], b["n"]) for b in own], "; "))
    push!(lines, @sprintf("  %-14s %10s %10s %7s %10s %13s %10s  %10s %8s  status", "group", "rate", "-> rate", "factor", "se (rate)", "jackknife se", "in k of n", "own rel se", "move/se"))
    jk = fit["jackknife"]
    for (g, name) in enumerate(fit["groups"])
        status = fit["identified"][g] ? "identified" : get(fit["held"], name, "")
        fit["cap_bound"][g] && (status *= ", at the cap")
        se = fit["se_rate"][g]; rel = fit["rel_se_marginal"][g]; mv = fit["moved_sigmas"][g]
        se_s = se === nothing ? lpad("-", 10) : @sprintf("%10.3g", se)
        rel_s = rel === nothing ? lpad("-", 10) : @sprintf("%10.2f", rel)
        mv_s = mv === nothing ? lpad("-", 8) : @sprintf("%8.1f", mv)
        if jk !== nothing
            jse = jk["se_rate"][g]
            jk_s = jse === nothing ? lpad("-", 13) : @sprintf("%13.3g", jse)
            in_s = @sprintf("%3d of %-3d", jk["identified_in"][g], jk["n_cooks"])
        else
            jk_s = lpad("-", 13); in_s = lpad("-", 10)
        end
        push!(lines, @sprintf("  %-14s %10.4g %10.4g %7.2f %s %s %10s  %s %s  %s [%s]", name, fit["rates"][g], fit["rates_new"][g], fit["factors"][g], se_s, jk_s, in_s, rel_s, mv_s, status, get(units, name, "")))
    end
    return join(lines, "\n")
end
"The lag table as the driver prints it."
function format_als_lags(rows)
    isempty(rows) && return "  (no lag pairs)"
    out = [@sprintf("  %-10s %7s %6s %8s %6s %13s %12s", "channel", "lag", "n", "sample", "+-se", "model before", "model after")]
    for r in rows
        push!(out, @sprintf("  %-10s %5.0f m %6d %8.3f %6.3f %13.3f %12.3f", r["channel"], r["lag_s"] / 60, r["n"], r["sample"], r["se"], r["model_before"], r["model_after"]))
    end
    return join(out, "\n")
end

# --- the calibration file of the factor-graph backend ---------------------------------------------------------------

"The factor-graph backend's calibration: its state's rates and its sensors' noise, as the bench earned them."
const SYSTEM_STATE_NOISE_FILE = joinpath(REPO_ROOT, "data", "calibration", "state_noise_system.json")

"""
    state_noise_record(bl, sn) -> Dict

The state noise a bench record and a calibration file carry, by the belief's noise groups as the accumulator
names them (`_als_group_name`: the filter's twelve names for one piece, `piece.field` beside the pit's for more):
each group's rate under its name and its initial sd under the name with `0`, the fields' structure, the node
jitter, the slab fields' smooth-mode count (`field_modes`, system_state.jl) and the events' disturbances. For one piece the keys
are `StateNoise`'s fields plus `field_modes`.
"""
function state_noise_record(bl::BeliefLayout, sn)
    d = Dict{String,Any}()
    for (g, _) in bl.groups
        nm = _als_group_name(bl, g)
        d[nm] = _group_rate(bl, sn, g, false); d[nm * "0"] = _group_rate(bl, sn, g, true)
    end
    ps = _pit_sn(sn)
    d["fields"] = string(ps.fields); d["node_jitter"] = ps.node_jitter
    d["field_modes"] = FIELD_MODES                                          # the slab fields' smooth modes beyond the bulk (system_state.jl)
    d["T_event"] = ps.T_event; d["T_spritz"] = ps.T_spritz; d["m_add"] = ps.m_add
    return d
end
"The unit of a noise group's rate, by its field's name."
_group_unit(g::AbstractString) = (f = String(last(rsplit(g, '.'; limit=2))); get(Dict("T" => "K", "W" => "kg/kg", "C" => "-", "F" => "-", "S" => "-", "bark" => "-", "T_pit" => "K",
                                                                                   "m_fuel" => "kg", "m_pan" => "kg", "T_pan" => "K", "T_fb" => "K", "ignited" => "-"), f, ""))
"""
    system_state_noise(record) -> StateNoise or Dict{Symbol,StateNoise}

The `StateNoise` a calibration record's rates describe: one for a one-piece record (the twelve names), a
dictionary by piece id with the pit's under `:pit` for a record whose groups carry piece prefixes (each piece's
`StateNoise` its own fields' rates and widths with the pit's shared ones).
"""
function system_state_noise(d::AbstractDict)
    rates = d["rates"]; init = d["init_sd"]
    common = Dict{Symbol,Any}()
    haskey(d, "fields") && (common[:fields] = Symbol(d["fields"]))
    haskey(d, "node_jitter") && (common[:node_jitter] = Float64(d["node_jitter"]))
    for (k, v) in get(d, "events", Dict()); common[Symbol(k)] = Float64(v); end
    names = collect(String, keys(rates))
    all(n -> n in STATE_GROUPS, names) && return state_noise_from_dict(d)
    pieces = unique([String(first(rsplit(n, '.'; limit=2))) for n in names if occursin('.', n)])
    pit_kw = Dict{Symbol,Any}()
    for n in names
        occursin('.', n) && continue
        pit_kw[Symbol(n)] = Float64(rates[n]); pit_kw[Symbol(n * "0")] = Float64(init[n])
    end
    out = Dict{Symbol,StateNoise}(:pit => StateNoise(; common..., pit_kw...))
    for id in pieces
        kw = copy(pit_kw)
        for n in names
            startswith(n, id * ".") || continue
            f = String(last(rsplit(n, '.'; limit=2)))
            kw[Symbol(f)] = Float64(rates[n]); kw[Symbol(f * "0")] = Float64(init[n])
        end
        out[Symbol(id)] = StateNoise(; common..., kw...)
    end
    return out
end
export state_noise_record, system_state_noise

"""
    system_calibration_record(noise, sensors; fit=nothing, bench=Dict(), method="") -> Dict

The file's contents from a `state_noise_record` dictionary: the rates and initial sds by group, the fields'
structure, the node jitter and the events' disturbances, per sensor id its white sd, OU sd and correlation
time, the units, the backend fingerprint (the physics and the backend's files: `backend_fingerprint`), the
physics fingerprint, the commit and the date, the method and the bench's numbers.
"""
function system_calibration_record(noise::AbstractDict, sensors::AbstractDict{Symbol,<:NamedTuple}; fit=nothing, bench=Dict{String,Any}(), method::AbstractString="")
    meta_keys = ("fields", "node_jitter", "field_modes", "T_event", "T_spritz", "m_add")
    groups = [String(k) for k in keys(noise) if !(String(k) in meta_keys) && !endswith(String(k), "0") && haskey(noise, String(k) * "0")]
    commit = try
        strip(read(`git -C $REPO_ROOT rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
    return Dict{String,Any}("backend" => "system", "rates" => Dict(g => Float64(noise[g]) for g in groups), "init_sd" => Dict(g => Float64(noise[g * "0"]) for g in groups),
                            "fields" => string(noise["fields"]), "node_jitter" => Float64(noise["node_jitter"]), "field_modes" => Int(get(noise, "field_modes", 0)),
                            "events" => Dict("T_event" => Float64(noise["T_event"]), "T_spritz" => Float64(noise["T_spritz"]), "m_add" => Float64(noise["m_add"])),
                            "sensors" => Dict(string(id) => Dict("white_sd" => nm.white_sd, "ar_sd" => nm.ar_sd, "tau" => nm.tau) for (id, nm) in sensors),
                            "units" => Dict(g => _group_unit(g) for g in groups),
                            "physics" => Dict("fingerprint" => backend_fingerprint(), "physics_fingerprint" => physics_fingerprint(), "commit" => commit,
                                              "date" => _today_string(),
                                              "what" => "BrisketEstimator.backend_fingerprint: src/**/*.jl, the filter's files and the factor-graph backend's; every driver refuses the file when the tree's differs, the tune round alone starts from stale rates"),
                            "method" => String(method), "fit" => fit, "bench" => bench)
end
_today_string() = Libc.strftime("%Y-%m-%d", time())
# JSON 1.8 refuses a NaN unless asked: the bench's records carry them (a consistency statistic that has none, a NEES
# whose block is singular), as the RxInfer bench's do, and the Python renderers read them; every writer and reader of
# the bench's files goes through these two
"A JSON document of `x` with NaN and infinities written as the tokens Python reads (`NaN`, `Infinity`); `pretty` indents."
json_document(x; pretty::Bool=false) = JSON.json(x; allownan=true, pretty)
"A JSON file parsed with NaN and infinities allowed."
parse_json_file(path::AbstractString) = JSON.parsefile(path; allownan=true)
export json_document, parse_json_file
write_system_calibration(path::AbstractString, record::AbstractDict) = open(path, "w") do io
    write(io, json_document(record; pretty=true))
end

"""
    load_system_calibration(path=SYSTEM_STATE_NOISE_FILE; strict=true, warn=true) -> (sn, sensors)

The backend's `StateNoise` (one, or a dictionary by piece: `system_state_noise`) and per sensor id
`(white_sd, ar_sd, tau)` as the bench earned them. Under `strict`
a missing file or a fingerprint that is not the tree's backend fingerprint is an error; with `strict=false` (the
tune round, which starts from the file by design) a missing file gives `nothing` for both with a note and a stale
file its numbers with a warning.
"""
function load_system_calibration(path::AbstractString=SYSTEM_STATE_NOISE_FILE; strict::Bool=true, warn::Bool=true)
    if !isfile(path)
        strict && error("no calibration file at $path: earn it with estimator/scripts/bench_system.jl, or take the defaults explicitly")
        warn && @info "no calibration file at $path: the StateNoise defaults and the matched sensor noise, a starting point and not a result"
        return nothing, nothing
    end
    d = parse_json_file(path)
    got = string(get(get(d, "physics", Dict()), "fingerprint", ""))
    fp = backend_fingerprint()
    if got != fp
        strict && error("the calibration in $path was earned on backend $got; the tree's is $fp: stale, rerun estimator/scripts/bench_system.jl (its tune round alone may start from it: strict=false)")
        warn && @warn "the calibration in $path was earned on backend $got; the tree's is $fp: stale, rerun estimator/scripts/bench_system.jl"
    end
    sn = system_state_noise(d)
    sensors = Dict{Symbol,NamedTuple}(Symbol(k) => (white_sd=Float64(v["white_sd"]), ar_sd=Float64(v["ar_sd"]), tau=Float64(v["tau"])) for (k, v) in get(d, "sensors", Dict()))
    return sn, sensors
end
