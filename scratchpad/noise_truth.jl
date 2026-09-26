using JSON, Statistics, Printf
m = JSON.parsefile(joinpath(@__DIR__, "..", "data", "mock", "mock_realcook.json"))
s = JSON.parsefile(joinpath(@__DIR__, "..", "data", "mock", "mock_realcook_stream.json"))
t = Float64.(m["t"]); dt = t[2]-t[1]; nc = m["noise"]
bs = s["sets"][1]["batches"]; last = bs[findlast(b -> !isempty(b["noise"]), bs)]; println("noise from batch at ", last["t_h"], " h")
for (ch, key) in [("pit","pit"), ("core","core"), ("humidity","humidity"), ("mass","mass")]
    y = Float64.(m["readings"][ch]); c = Float64.(m["clean"][ch]); mk = Bool.(m["masks"][ch])
    ok = ch == "humidity" ? mk .& (y .< nc["hum_cap"] - 1e-9) : mk
    e = (y .- c)[ok]
    idx = findall(ok); g = minimum(diff(idx))
    e1 = e[1:end-1]; e2 = e[2:end]; ρg = cor(e1, e2)
    n = last["noise"][key]
    @printf("%-9s empirical sd %.4g  ρ per sample (spacing %2d steps) %.3f -> per step %.3f | stream final σ %.4g ρ/step %.3f | prior file %s\n",
            ch, std(e), g, ρg, ρg^(1/g), n["σ"], n["ρ"], s["noise_prior"][key])
end
println("noise_truth in record: ", s["noise_truth"])
println("mock noise: ", Dict(k => v for (k, v) in nc if occursin("sd", k) || occursin("tau", k)))
