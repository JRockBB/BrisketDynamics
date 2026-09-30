using Profile
function profile_reference()
    case=first(ONLINE_COMPARISON_CASES)
    model=consistency_model(nx=4,suite=case.suite,stop=maximum(case.times))
    batches,_=comparison_readings(model,case)
    inner=online_init(model;lag=typemax(Int))
    E._append_sampled_measurements!(inner,reduce(vcat,batches))
    ref=E._posterior_transport(inner)
    rng=MersenneTwister(5401);U=randn(rng,ref.n,512)
    U=ref.center.+U+ref.V*((ref.a.-1).*(ref.V'*U))
    for j in 1:4;E._transport_draw(ref,view(U,:,j));end
    Profile.clear();start=time()
    @profile for j in axes(U,2);E._transport_draw(ref,view(U,:,j));end
    wall=time()-start
    open(joinpath(@__DIR__,"transport_profile.txt"),"w") do io
        println(io,"Backend ",BASE_BACKEND,"; 512 full-history transport evaluations; wall=",wall)
        Profile.print(io;format=:flat,sortedby=:count,mincount=10)
    end
    println("PROFILE COMPLETE wall=",wall);flush(stdout)
    return ref,U
end
profile_ref,profile_coordinates=profile_reference()
