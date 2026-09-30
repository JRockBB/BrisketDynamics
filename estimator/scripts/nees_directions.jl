# Where the Gaussian estimator's state error score comes from, direction by direction, on the matched control.
# At every arrival the belief error is resolved along the eigen-directions of the reported covariance: each
# direction's claimed sd, the absolute error along it, its share of the score and its group loadings. The score is
# also formed on the reduced state (each field's node mean, every scalar as is), which drops the field-shape
# directions whose process noise is the tiny per-node jitter.
# Run: julia --threads=4 --project=estimator estimator/scripts/nees_directions.jl [n=8] [seed0=3001] [suite=full] [nx=4] [stop=1800] [lag=full]
include(joinpath(@__DIR__, "consistency_control.jl"))

"The rows that reduce a belief vector to one entry per group: a field group's node mean, a scalar group's value."
function reduction_matrix(bl)
    rows = Vector{Float64}[]; names = String[]
    for (g, rg) in bl.groups
        if length(rg) > 1 && !endswith(g, ".bark")
            r = zeros(bl.n); r[rg] .= 1 / length(rg); push!(rows, r); push!(names, g * ".mean")
        else
            for i in rg; r = zeros(bl.n); r[i] = 1; push!(rows, r); push!(names, length(rg) > 1 ? "$g[$(i-first(rg)+1)]" : g); end
        end
    end
    return reduce(vcat, permutedims.(rows)), names
end

function group_of(bl, i)
    for (g, rg) in bl.groups; i in rg && return g; end
    return "?"
end

function direction_record(r, truth, bl, A)
    e = r.y - truth.y; P = Symmetric(r.y_cov)
    F = eigen(P); λ = max.(F.values, eps()); V = F.vectors
    a = V' * e; c2 = a .^ 2 ./ λ
    total = sum(c2)
    order = sortperm(c2; rev=true)
    top = map(order[1:3]) do k
        load = Dict{String,Float64}()
        for i in 1:bl.n; g = group_of(bl, i); load[g] = get(load, g, 0.0) + V[i, k]^2; end
        main = sort(collect(load); by=last, rev=true)[1:2]
        (sd=sqrt(λ[k]), err=abs(a[k]), share=c2[k] / total, groups=[(g, round(w; digits=2)) for (g, w) in main])
    end
    er = A * e; Pr = Symmetric(A * r.y_cov * A')
    reduced = dot(er, Pr \ er) / length(er)
    # the reduced state's worst entries, standardized
    zr = er ./ sqrt.(diag(Pr))
    return (nees=total / bl.n, reduced=reduced, top=top, zr=zr,
            small_share=sum(c2[sqrt.(λ) .< 0.01 * maximum(sqrt.(λ))]) / total)
end

function run_directions(; n=8, seed0=3001, suite=:full, nx=4, stop=1800.0, lag=typemax(Int))
    model = consistency_model(; nx, suite, stop)
    bl = model.bl; A, rnames = reduction_matrix(bl)
    times = collect(0.0:120.0:stop)
    all = Dict{Float64,Vector{Any}}()
    for seed in seed0:seed0+n-1
        t0 = time()
        recs = Any[]
        control_cook(model, seed; times, lag, inspect=(sm, r, truth) -> push!(recs, direction_record(r, truth, bl, A)))
        for (t, rec) in zip(times, recs); push!(get!(all, t, Any[]), rec); end
        last = recs[end]
        @printf("seed %d (%.0fs): state score/dim %.2f, reduced/dim %.2f, share in directions under 1%% of the widest sd %.3f\n",
                seed, time() - t0, last.nees, last.reduced, last.small_share)
        for d in last.top
            @printf("    sd %.2e err %.2e share %.2f %s\n", d.sd, d.err, d.share, join(["$g:$w" for (g, w) in d.groups], " "))
        end
        worst = sortperm(abs.(last.zr); rev=true)[1:4]
        println("    reduced worst: ", join(["$(rnames[i])=$(round(last.zr[i]; digits=2))" for i in worst], "  "))
        flush(stdout)
    end
    println("\nmean over cooks by time: full-state score/dim | reduced-state score/dim")
    for t in times
        rs = all[t]
        @printf("t %5.0f  %8.2f  %6.2f\n", t, mean(x.nees for x in rs), mean(x.reduced for x in rs))
    end
    # the reduced state's standardized error per entry at the end, across cooks (bias and rms)
    Z = reduce(hcat, [x.zr for x in all[times[end]]])
    println("\nreduced entries at t=$(times[end]): rms of standardized error across cooks (1 is consistent)")
    for i in sortperm(vec(sqrt.(mean(Z .^ 2; dims=2))); rev=true)[1:min(12, length(rnames))]
        @printf("  %-22s rms %.2f  mean %+.2f\n", rnames[i], sqrt(mean(Z[i, :] .^ 2)), mean(Z[i, :]))
    end
    return all
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = ARGS
    run_directions(n=length(a) > 0 ? parse(Int, a[1]) : 8, seed0=length(a) > 1 ? parse(Int, a[2]) : 3001,
                   suite=length(a) > 2 ? Symbol(a[3]) : :full, nx=length(a) > 3 ? parse(Int, a[4]) : 4,
                   stop=length(a) > 4 ? parse(Float64, a[5]) : 1800.0,
                   lag=length(a) > 5 && a[6] != "full" ? parse(Int, a[6]) : typemax(Int))
end
