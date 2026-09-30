function _weak_moves!(ref,U,X,ratio,beta,mixture_density,rng,moves)
    # Include latent as well as state directions in the weak complement.
    A=ref.V[:,ref.a.<.99];B=if isempty(A)
        zeros(ref.n,0)
    else
        F=svd(A;full=false);F.U[:,F.S.>1e-10]
    end
    size(B,2)==ref.n && return 0
    root(v)=v+ref.V*((ref.a.-1).*(ref.V'*v))
    invroot(v)=v+ref.V*((1 ./ ref.a.-1).*(ref.V'*v))
    n=size(U,2);seeds=rand(rng,UInt64,n);counts=zeros(Int,n)
    Threads.@threads for j in 1:n
        localrng=MersenneTwister(seeds[j]);u=copy(U[:,j]);r=ratio[j]
        for move in 1:moves
            global_component=rand(localrng)<mixture_density(u)[2]
            eta=global_component ? copy(u) : invroot(u-ref.center)
            fixed=B*(B'*eta);weak=eta-fixed;nu=randn(localrng,ref.n);nu-=B*(B'*nu)
            level=beta*r+log(rand(localrng));angle=2pi*rand(localrng);lo=angle-2pi;hi=angle
            accepted=false
            for attempt in 1:512
                candidate_eta=fixed+weak*cos(angle)+nu*sin(angle)
                candidate=global_component ? candidate_eta : ref.center+root(candidate_eta)
                result=_transport_draw(ref,candidate);counts[j]+=1
                isfinite(result.potential) && all(isfinite,result.draw) || error("non-finite weak-block proposal")
                candidate_r=-result.potential-mixture_density(candidate)[1]
                if beta*candidate_r>=level
                    u=candidate;r=candidate_r;X[:,j]=result.draw;accepted=true;break
                end
                if angle<0;lo=angle;else;hi=angle;end
                angle=lo+rand(localrng)*(hi-lo)
            end
            accepted || error("weak-block slice failed")
        end
        U[:,j]=u;ratio[j]=r
    end
    return sum(counts)
end

