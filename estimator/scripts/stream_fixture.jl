# Shared setup for the streaming bench and its truth/derivative diagnostics.
# The lighting design is an explicitly independent-marginal validation prior,
# not a replacement for the calibration chain or the production next-cook prior.
function stream_fixture(repo; start::Symbol=:legacy)
    start in (:legacy, :lighting) || throw(ArgumentError("STREAM_START is legacy or lighting"))
    m = JSON.parsefile(joinpath(repo, "data", "mock", "mock_realcook.json"))
    t = Float64.(m["t"]); u = Float64.(m["u"]); meta = m["meta"]
    cfg = MeatConfig(; T0=Float64(meta["tcore0_K"]))
    old = latent_block(joinpath(repo, "data", "calibration", "prior_next_cook.json"), cfg)
    sites, names = old.prior_phys, old.phys_names
    ev = m["events"]
    load = BrisketDynamics.LOAD_AT_START
    if start === :lighting
        fit = JSON.parsefile(joinpath(repo, "data", "calibration", "gn_map_real.json"))
        fitted = JSON.parsefile(joinpath(repo, "data", "calibration", "map_real.json"))
        load = Float64(fit["preheat_s"])
        step = t[2] - t[1]
        npre = round(Int, load / step)
        isapprox(npre * step, load; atol=1e-6) || error("preheat must align with the mock control grid")
        t = collect(0.0:step:(last(t) + load))
        u = vcat(fill(Float64(fitted["u_pre"]), npre), u)
        sites = map(names) do name
            x = Float64(fitted[name])
            if name == "a0"
                # A chimney start: mean 0.15, sd 0.05. This spread is a declared
                # design assumption; a0 was fixed, not estimated, in the refit.
                concentration = x * (1 - x) / 0.05^2 - 1
                PriorSite("beta", x * concentration, (1 - x) * concentration; coord=:score)
            elseif name in ("m_conv", "u_leak")
                s = Float64(fit["sd_z_fisher"][name])
                concentration = 1 / (s^2 * x * (1 - x)) - 1
                PriorSite("beta", x * concentration, (1 - x) * concentration; coord=:score)
            else
                PriorSite("lognormal", log(x), Float64(fit["sd_z_fisher"][name]))
            end
        end
    end
    shift = start === :lighting ? load : 0.0
    events = (t_load=load, t_wrap=Float64(ev["t_wrap"]) + shift, t_pull=Float64(ev["t_pull"]) + shift,
              t_add=Float64(ev["t_add"]) + shift, t_spritz=Float64(ev["t_spritz"]) + shift,
              dt_spritz=Float64(ev["dt_spritz"]), m_spritz=Float64(ev["m_spritz"]))
    pit = PitConfig(; T_amb=Float64(meta["t_amb_K"]),
                    Tpit0=Float64(meta[start === :lighting ? "t_amb_K" : "tpit0_K"]), m0=Float64(meta["m0_kg"]))
    return (; t, u, pit, cfg, sites, names, events)
end
