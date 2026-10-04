using Test, JSON

const SCORE_SCRIPT = joinpath(@__DIR__, "..", "scripts", "stream_validate.jl")

@testset "failed and incomplete streaming cooks cannot pass validation" begin
    mktempdir() do dir
        record = Dict{String,Any}("cooks" => [], "skipped_seeds" => [41002],
            "failures" => [Dict("seed" => 41002, "gradient" => NaN, "newton_sd" => Inf)],
            "prior_relin" => Inf, "backend_fingerprint" => "test-only")
        input = joinpath(dir, "failed.json")
        output = joinpath(dir, "score.json")
        save() = write(input, JSON.json(record; allownan=true))
        cmd = `$(Base.julia_cmd()) --threads=1 --project=$(joinpath(@__DIR__, "..")) $SCORE_SCRIPT $output $input`
        save()
        @test success(pipeline(cmd; stdout=devnull, stderr=devnull))
        score = JSON.parsefile(output; allownan=true)
        @test score["n_cooks"] == 0
        @test !score["checkpoint_gates_pass"] && !score["all_reported_gates_pass"]
        @test score["skipped_seeds"] == [41002]
        @test isnan(only(score["failures"])["gradient"])
        @test isinf(only(score["failures"])["newton_sd"])
        record["active_cook"] = Dict("seed" => 41003)
        save()
        rm(output)
        @test !success(pipeline(cmd; stdout=devnull, stderr=devnull))
        @test !isfile(output)
    end
end

@testset "independent-cook intervals and per-site gates" begin
    mktempdir() do dir
        batch = Dict("t_h" => 2.0, "state_t_h" => 2.0, "err_sd" => [10.0],
            "nees_z" => 1.0, "nees_T" => 3.0, "nees_g" => Dict("T_pit" => 100.0))
        row = Dict("committed" => true, "converged" => true, "dof" => 1, "n_readings" => 2, "nis" => 1.0,
            "innovation_whitened" => [])
        record = Dict("names" => ["test_latent"], "groups" => ["T_pit"], "skipped_seeds" => [],
            "cooks" => [Dict("seed" => seed, "batches" => [batch], "temps" => [row]) for seed in 1:4])
        input, output = joinpath(dir, "synthetic.json"), joinpath(dir, "score.json")
        write(input, JSON.json(record))
        old_args = copy(ARGS)
        try
            empty!(ARGS); append!(ARGS, [output, input])
            include(SCORE_SCRIPT)
        finally
            empty!(ARGS); append!(ARGS, old_args)
        end
        score = JSON.parsefile(output; allownan=true)
        edge = only(score["checkpoints"])
        @test edge["latents"]["pass"] && edge["temperatures"]["pass"]
        @test !edge["per_site"]["test_latent"]["pass"]
        @test !edge["state_groups"]["T_pit"]["pass"]
        @test !score["checkpoint_gates_pass"] && !score["all_reported_gates_pass"]
        @test score["n_censored"] == 4 && !score["innovation_gaussian_null_valid"]
    end
end

@testset "analytic statistical reference values" begin
    @test chi_quantile(0.975, 2) ≈ -2log(0.025) rtol=1e-12
    @test t_critical(1) ≈ tan(0.475pi) rtol=1e-12
    @test t_critical(2) ≈ sqrt(2 * 0.95^2 / (1 - 0.95^2)) rtol=1e-12
    @test !cluster_score([1.0], 1.0)["pass"]
    @test cluster_score([0.9, 1.1, 1.0, 1.0], 1.0)["pass"]
    @test !cluster_score([10.0, 10.1, 9.9], 1.0)["pass"]
    @test !cluster_score([nothing, 1.0], 1.0)["pass"]
    @test !site_score([1.0])["pass"]
end
