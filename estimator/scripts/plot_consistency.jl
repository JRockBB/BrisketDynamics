# The figures of the 2026-09-30 consistency diagnosis (docs/results.md), read from the diagnostic logs in
# scratchpad/claude_consistency/ (nees_directions.jl, forecast_check.jl, trace_pan.jl, the depth profiles).
# Run from the repo root in its own CairoMakie environment: julia --project=estimator/scripts/plotenv estimator/scripts/plot_consistency.jl
using CairoMakie, Printf, Statistics

const LOGS = joinpath(@__DIR__, "..", "..", "scratchpad", "claude_consistency")
const OUT = joinpath(@__DIR__, "..", "..", "figures", "estimator")
readlog(f) = readlines(joinpath(LOGS, f))

# the validated categorical order (dataviz reference palette, light surface)
const C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
const MARK = [:circle, :rect, :diamond, :utriangle, :dtriangle]
const INK = "#0b0b0b"; const INK2 = "#52514e"; const GRID = "#e4e3df"; const SURF = "#fcfcfb"
const QN = ["T_pit", "T_centre", "T_surface", "T_mean", "W_mean"]
const QLABEL = ["pit temperature", "centre temperature", "surface temperature", "mean meat temperature", "mean meat water"]

set_theme!(Theme(fontsize=14, backgroundcolor=SURF, textcolor=INK,
    Axis=(backgroundcolor=SURF, xgridcolor=GRID, ygridcolor=GRID, xtickcolor=INK2, ytickcolor=INK2,
          bottomspinecolor=INK2, leftspinecolor=INK2, topspinevisible=false, rightspinevisible=false,
          xlabelcolor=INK2, ylabelcolor=INK2, titlealign=:left, titlesize=15),
    Legend=(framevisible=false, backgroundcolor=SURF)))

# ---- parsers -------------------------------------------------------------------------------------------------
function directions(f)
    t = Float64[]; full = Float64[]; red = Float64[]
    for l in readlog(f)
        m = match(r"^t\s+(\d+)\s+([\d.]+)\s+([\d.]+)$", strip(l))
        m === nothing && continue
        push!(t, parse(Float64, m[1]) / 60); push!(full, parse(Float64, m[2])); push!(red, parse(Float64, m[3]))
    end
    return t, full, red
end
function forecast_summary(f)
    d = Dict{Tuple{Int,String},NTuple{3,Float64}}()
    for l in readlog(f)
        m = match(r"\+\s*(\d+) min (\S+)\s+rms z ([\d.]+)\s+mean z ([+-][\d.]+)\s+outside 90% ([\d.]+)", l)
        m === nothing && continue
        d[(parse(Int, m[1]), m[2])] = (parse(Float64, m[3]), parse(Float64, m[4]), parse(Float64, m[5]))
    end
    return d
end
function per_cook(f)
    d = Dict{Int,Dict{String,Float64}}()
    for l in readlog(f)
        m = match(r"^seed (\d+) \(.*z-scores: (.*)$", l)
        m === nothing && continue
        d[parse(Int, m[1])] = Dict(String(split(kv, '=')[1]) => parse(Float64, split(kv, '=')[2]) for kv in split(m[2]))
    end
    return d
end
function profile(f)
    k = Float64[]; dF = Float64[]; head = ""
    for l in readlog(f)
        startswith(l, "latent") && (head = l)
        m = match(r"k=([-\d.]+): ΔF ([-\d.]+)", l)
        m === nothing && continue
        push!(k, parse(Float64, m[1])); push!(dF, parse(Float64, m[2]))
    end
    return k, dF, head
end

# ---- 1: where the full-state score lives ---------------------------------------------------------------------
let fig = Figure(size=(1100, 430))
    for (j, (f, ttl)) in enumerate((("directions_full.log", "Before: truths drawn with the old noise"),
                                     ("directions_truthdamped.log", "After: truths drawn as physical noise")))
        t, full, red = directions(f)
        ax = Axis(fig[1, j]; title=ttl, xlabel="minutes into the cook", ylabel=j == 1 ? "error score per dimension (1 is consistent)" : "",
                  yscale=log10, yticks=[0.5, 1, 2, 5, 10, 20], limits=(nothing, (0.5, 25)))
        hlines!(ax, [1.0]; color=INK2, linestyle=:dash, linewidth=1)
        lines!(ax, t, full; color=C[1], linewidth=2); scatter!(ax, t, full; color=C[1], marker=MARK[1], markersize=9)
        lines!(ax, t, red; color=C[2], linewidth=2); scatter!(ax, t, red; color=C[2], marker=MARK[2], markersize=9)
        text!(ax, 14.0, 17.0; text="full state", color=INK, align=(:center, :bottom))
        text!(ax, 14.0, 0.62; text="field means + scalars", color=INK, align=(:center, :bottom))
    end
    Label(fig[0, :], "The large scores sit in the meat fields' node-to-node shape; field means and scalars stay near 1 (8 matched cooks, full suite)";
          fontsize=16, halign=:left, tellwidth=false)
    save(joinpath(OUT, "consistency_score_directions.png"), fig; px_per_unit=2)
end

# ---- 2: forecast calibration by horizon and suite ------------------------------------------------------------
let fig = Figure(size=(1300, 720))
    runs = (("forecast_full.log", "Full suite, old truths"), ("forecast_truthdamped.log", "Full suite"),
            ("forecast_core.log", "One core probe"), ("forecast_two_core.log", "Two core probes (15 cooks)"))
    hz = [30, 60, 120]
    axes_ = []
    for (j, (f, ttl)) in enumerate(runs)
        d = forecast_summary(f)
        a1 = Axis(fig[1, j]; title=ttl, xticks=hz, ylabel=j == 1 ? "rms of standardized error\n(1 is calibrated)" : "",
                  limits=((20, 130), (0.3, 2.1)))
        a2 = Axis(fig[2, j]; xticks=hz, xlabel="forecast horizon, minutes", ylabel=j == 1 ? "share of truths outside\nthe 90% band (0.10)" : "",
                  limits=((20, 130), (-0.02, 0.32)))
        hlines!(a1, [1.0]; color=INK2, linestyle=:dash, linewidth=1)
        hlines!(a2, [0.10]; color=INK2, linestyle=:dash, linewidth=1)
        for (i, q) in enumerate(QN)
            r = [d[(h, q)][1] for h in hz]; o = [d[(h, q)][3] for h in hz]
            scatterlines!(a1, hz, r; color=C[i], marker=MARK[i], markersize=10, linewidth=2)
            scatterlines!(a2, hz, o; color=C[i], marker=MARK[i], markersize=10, linewidth=2)
        end
        j > 1 && (hideydecorations!(a1; grid=false, ticks=false, ticklabels=false); hideydecorations!(a2; grid=false, ticks=false, ticklabels=false))
        push!(axes_, a1)
    end
    Legend(fig[3, :], [[LineElement(color=C[i], linewidth=2), MarkerElement(color=C[i], marker=MARK[i], markersize=10)] for i in 1:5],
           QLABEL; orientation=:horizontal, tellheight=true)
    Label(fig[0, :], "Forecast calibration from 30 min into the cook, 16 matched cooks per suite: calibrated once the truths are physical; two probes run about 20% narrow";
          fontsize=16, halign=:left, tellwidth=false)
    save(joinpath(OUT, "consistency_forecast_calibration.png"), fig; px_per_unit=2)
end

# ---- 3: every cook's 1-h forecast error ----------------------------------------------------------------------
let fig = Figure(size=(1300, 470))
    runs = (("forecast_full.log", "Full suite, old truths"), ("forecast_truthdamped.log", "Full suite"),
            ("forecast_core.log", "One core probe"), ("forecast_two_core.log", "Two core probes (15 cooks)"))
    for (j, (f, ttl)) in enumerate(runs)
        d = per_cook(f); seeds = sort(collect(keys(d)))
        ax = Axis(fig[1, j]; title=ttl, xticks=(1:5, ["pit", "centre", "surface", "mean T", "water"]),
                  ylabel=j == 1 ? "standardized 1-h forecast error" : "", limits=((0.4, 5.6), (-5, 5)))
        band!(ax, [0.4, 5.6], [-1.645, -1.645], [1.645, 1.645]; color=(C[1], 0.10))
        hlines!(ax, [0.0]; color=INK2, linewidth=1)
        for (i, q) in enumerate(QN)
            z = [d[s][q] for s in seeds]
            x = i .+ range(-0.25, 0.25; length=length(z))
            scatter!(ax, x, z; color=C[i], marker=MARK[i], markersize=9, strokecolor=SURF, strokewidth=1)
            for (xx, zz, s) in zip(x, z, seeds)
                abs(zz) > 2.4 && text!(ax, xx, zz; text=string(s), fontsize=11, color=INK2, align=(:left, :center), offset=(5, 0))
            end
        end
        j > 1 && hideydecorations!(ax; grid=false, ticks=false, ticklabels=false)
    end
    Label(fig[0, :], "Each cook's 1-h forecast error; the shaded band is the central 90% a calibrated forecast would hold (seeds labeled beyond 2.4)";
          fontsize=16, halign=:left, tellwidth=false)
    save(joinpath(OUT, "consistency_forecast_per_cook.png"), fig; px_per_unit=2)
end

# ---- 4: the probe-depth profiles ------------------------------------------------------------------------------
let fig = Figure(size=(1100, 480))
    for (j, (f, ttl)) in enumerate((("profile_3012_17.log", "First probe's depth"), ("profile_3012.log", "Second probe's depth")))
        k, dF, _ = profile(f)
        kk = range(-4.2, 3.2; length=200)
        ax = Axis(fig[1, j]; title=ttl, xlabel="depth pinned at the estimate + k reported sd", ylabel=j == 1 ? "rise of the objective" : "",
                  limits=((-4.3, 3.3), (-0.5, 18)))
        lines!(ax, kk, 0.5 .* kk .^ 2; color=INK2, linestyle=:dash, linewidth=2)
        scatterlines!(ax, vcat(k[1:4], 0.0, k[5:end]), vcat(dF[1:4], 0.0, dF[5:end]); color=C[1], marker=MARK[1], markersize=10, linewidth=2)
        text!(ax, 1.2, 13.0; text="solid: the objective", color=INK, align=(:center, :bottom))
        text!(ax, 1.2, 11.5; text="dashed: a Gaussian with the reported width", color=INK2, align=(:center, :bottom))
    end
    Label(fig[0, :], "Two-probe cook 3012 at 10 min, each depth pinned and everything else re-solved: the objective rises far
slower than a Gaussian on the side toward the cold centre, so the reported depth width is too narrow there";
          fontsize=16, halign=:left, tellwidth=false)
    save(joinpath(OUT, "consistency_depth_profile.png"), fig; px_per_unit=2)
end

# ---- 5: the absent pan's truth; the depth bank against the Gaussian ------------------------------------------
let fig = Figure(size=(1200, 450))
    t = Float64[]; truth = Float64[]; est = Float64[]; on = false
    for l in readlog("trace_pan.log")
        startswith(l, "seed 3006") && (on = true; continue)
        startswith(l, "seed") && (on = false)
        on || continue
        m = match(r"^\s*(\d+) \| ([\d.eE+-]+)\s+([\d.eE+-]+)", l)
        m === nothing && continue
        push!(t, parse(Float64, m[1]) / 60); push!(truth, parse(Float64, m[2])); push!(est, parse(Float64, m[3]))
    end
    ax = Axis(fig[1, 1]; title="Cook 3006 with no water pan configured, old truths", xlabel="minutes into the cook", ylabel="pan water, kg")
    scatterlines!(ax, t, truth; color=C[2], marker=MARK[2], markersize=9, linewidth=2)
    scatterlines!(ax, t, est; color=C[1], marker=MARK[1], markersize=9, linewidth=2)
    text!(ax, t[argmax(truth)], maximum(truth); text="the simulated truth", color=INK, align=(:right, :center), offset=(-8, 0))
    text!(ax, 1.0, est[1]; text="the belief (no pan)", color=INK, align=(:left, :bottom), offset=(0, 8))
    g = per_cook("forecast_two_core.log"); b = per_cook("forecast_two_core_depth.log")
    seeds = sort([s for s in keys(b) if haskey(g, s)])
    ax2 = Axis(fig[1, 2]; title="Two probes, the same $(length(seeds)) cooks: 1-h forecast errors", xlabel="Gaussian backend", ylabel="depth bank",
               limits=((-3.6, 3.6), (-3.6, 3.6)), aspect=DataAspect())
    lines!(ax2, [-3.6, 3.6], [-3.6, 3.6]; color=INK2, linestyle=:dash, linewidth=1)
    for (i, q) in enumerate(QN)
        scatter!(ax2, [g[s][q] for s in seeds], [b[s][q] for s in seeds]; color=C[i], marker=MARK[i], markersize=11, strokecolor=SURF, strokewidth=1)
    end
    Legend(fig[1, 3], [MarkerElement(color=C[i], marker=MARK[i], markersize=11) for i in 1:5], QLABEL)
    Label(fig[0, :], "Left: the old truth generator grew an absent pan to 2.4 kg. Right: the depth bank does not beat the Gaussian (points off the diagonal either way)";
          fontsize=16, halign=:left, tellwidth=false)
    save(joinpath(OUT, "consistency_pan_and_bank.png"), fig; px_per_unit=2)
end
println("wrote 5 figures to ", normpath(OUT))
