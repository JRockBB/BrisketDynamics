using Test
isdefined(@__MODULE__,:target_state_fingerprint) || include(joinpath(@__DIR__,"../scripts/target_ownership.jl"))

@testset "target ownership with captured data" begin
    let captured=[2.]
        fn=x->x+captured[1]
        target=(callable=fn,shared=captured);before=target_state_fingerprint(target)
        @test target_state_fingerprint(target)==before
        clone=deepcopy(target)
        @test !isequal(target.callable,clone.callable)
        @test target_state_fingerprint(clone)==before
        clone.shared[1]=3.
        @test clone.callable(0.)==3.
        @test target.callable(0.)==2.
        @test target_state_fingerprint(target)==before
        @test target_state_fingerprint(clone)!=before
        captured[1]=4.
        @test target_state_fingerprint(target)!=before
    end
end
