"""
Power-of-two Sobol nets with independent linear matrix scrambles and digital shifts.
The origin is retained and scrambled along with every other point. Error estimates
must use independent nets, not pretend the points within a net are IID.
"""
module QMCNormalRules
using Sobol, Random, SpecialFunctions
export uniform_net, normal_net
const BITS=53
const MASK=(UInt64(1)<<BITS)-1

function uniform_net(n::Int,dimension::Int;rng::AbstractRNG=MersenneTwister(0),scramble=true)
    n>=2 && ispow2(n) && n<=2^30 && dimension>0 || throw(ArgumentError("use a positive dimension and a power-of-two net size between 2 and 2^30"))
    depth=trailing_zeros(n);sequence=Sobol.SobolSeq(dimension)
    directions=zeros(UInt64,dimension,depth)
    for k in 1:depth,i in 1:dimension
        # Sobol 1.5 stores the direction numerator m[i,k] in base two.
        # The independent tests also compare the unscrumbled path against next!.
        directions[i,k]=UInt64(sequence.m[i,k])<<(BITS-k)
    end
    shift=zeros(UInt64,dimension)
    if scramble
        for i in 1:dimension
            rows=Vector{UInt64}(undef,BITS)
            for b in 1:BITS
                diagonal=UInt64(1)<<(BITS-b)
                earlier=MASK ⊻ ((diagonal<<1)-1)
                rows[b]=(rand(rng,UInt64)&earlier)|diagonal
            end
            for k in 1:depth
                v=directions[i,k];out=UInt64(0)
                for b in 1:BITS
                    out|=UInt64(isodd(count_ones(v&rows[b])))<<(BITS-b)
                end
                directions[i,k]=out
            end
            shift[i]=rand(rng,UInt64)&MASK
        end
    end
    current=copy(shift);points=Matrix{Float64}(undef,dimension,n)
    for j in 1:n
        if j>1
            k=trailing_zeros(j-1)+1
            for i in 1:dimension;current[i]⊻=directions[i,k];end
        end
        for i in 1:dimension
            # A half-ULP midpoint avoids infinities when mapping to a normal.
            # Its finite-resolution integration bias is not an error certificate.
            value=(Float64(current[i])+(scramble ? .5 : 0.))*2.0^-BITS
            points[i,j]=scramble ? clamp(value,nextfloat(0.),prevfloat(1.)) : value
        end
    end
    points
end

function normal_net(n::Int,dimension::Int;kwargs...)
    U=uniform_net(n,dimension;kwargs...)
    all(x->0<x<1,U) || throw(ArgumentError("normal nets require an interior-point scramble"))
    -sqrt(2).*erfcinv.(2U)
end
end