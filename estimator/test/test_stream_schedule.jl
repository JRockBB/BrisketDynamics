using Test
include(joinpath(@__DIR__,"../scripts/stream_schedule.jl"))

@testset "arrival scheduling preserves causality and the compute backlog" begin
    recs = [(t=t,) for t in (1.,3.,4.,4.,6.,10.)]
    @test stream_arrival_batch(recs,1,0.;paced=true) == (1,1.)
    # First solve takes 4 s: readings at 3 and 4 wait, but 6 has not arrived.
    @test stream_arrival_batch(recs,2,5.;paced=true) == (4,5.)
    # A 0.5 s solve finishes at 5.5, so the idle estimator waits only to 6.
    @test stream_arrival_batch(recs,5,5.5;paced=true) == (5,6.)
    @test stream_arrival_batch(recs,2,5.;paced=false) == (2,3.)
    @test stream_arrival_batch(recs,3,100.;paced=false) == (4,4.)
    rows = [Dict("committed"=>true,"processing_finish_s"=>5.),
            Dict("committed"=>true,"processing_finish_s"=>5.5),
            Dict("committed"=>false,"processing_finish_s"=>6.)]
    @test stream_available_at(rows,4.) === nothing
    @test stream_available_at(rows,5.) == 1
    @test stream_available_at(rows,10.) == 2
end
