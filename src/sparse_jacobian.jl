# A sparse Jacobian for systems of grid pieces. A dense Jacobian of a 3-D piece (5 fields per node, 3 states per
# exposed face) is out of reach past a few thousand nodes; the physics is local, but for two global scalars: the
# chamber's resolved vapor pressure (every exposed node's evaporation reads it, every exposed node feeds it) and each
# piece's exposed-area factor (every node's water and fat feed it). The Jacobian is therefore taken of the system's
# right-hand side with those two scalars held at their current values (`freeze`), on the pattern that local physics
# leaves: each node's fields against its own and its edge neighbours', each exposed face's bark, smoke and ring
# against its node (the ring against its inward line), the exposed nodes and the faces' states against the pit's
# states, and the pit against itself (the meat's heat drawn from the pit, one dense row, left out). The solve uses the
# exact right-hand side; the BDF's Newton iteration tolerates the approximate Jacobian (the tolerances govern the
# error). The columns are coloured so no two of one colour share a row (greedy distance-2), and each pass of the
# forward-mode evaluation seeds `chunk` colours at once.

struct _JacTag end

# the system's sparsity pattern (rows, columns) when every piece is a grid piece
function _grid_system_pattern(sys::CookSystem, ix)
    I = Int[]; J = Int[]
    add(r, c) = (push!(I, r); push!(J, c))
    pit = collect(ix.pit)
    for (k, pc) in enumerate(sys.pieces)
        pc isa GridPiece || throw(ArgumentError("the sparse Jacobian takes grid pieces only; $(pc.id) is a $(typeof(pc))"))
        o = first(ix.x[k]) - 1
        mesh = pc.mesh; N = mesh.N; nb = length(mesh.boundary)
        nbrs = [Int[] for _ in 1:N]
        for (a, b, _) in mesh.edges
            push!(nbrs[a], b); push!(nbrs[b], a)
        end
        fld(f, n) = o + (f - 1) * N + n                               # T, W, C, F, S of node n
        for n in 1:N, m in vcat(n, nbrs[n]), r in 1:5, c in 1:5
            add(fld(r, n), fld(c, m))
        end
        bs(f, b) = o + 5N + (f - 1) * nb + b                          # bark, smoke, ring of face b
        for (b, (n, _)) in enumerate(mesh.boundary)
            for r in (fld(1, n), fld(2, n)), c in pit                 # the exposed node's balance against the pit
                add(r, c)
            end
            for f in 1:3
                add(bs(f, b), bs(f, b)); add(bs(f, b), fld(1, n)); add(bs(f, b), fld(2, n)); add(bs(f, b), fld(5, n))
                for c in pit
                    add(bs(f, b), c)
                end
            end
            for m in mesh.inward[b]                                   # the ring's front reads its inward line
                add(bs(3, b), fld(1, m))
            end
        end
    end
    for r in pit, c in pit
        add(r, c)
    end
    S = sparse(I, J, ones(length(I)), sys.layout.n, sys.layout.n)
    return S
end

# greedy distance-2 colouring of the columns: no two columns of one colour share a row
function _colour_columns(S::SparseMatrixCSC)
    n = size(S, 2)
    Rt = sparse(S')                                                   # the rows' columns
    colour = zeros(Int, n); forbidden = zeros(Int, n + 1)
    for j in 1:n
        for k in nzrange(S, j)
            i = S.rowval[k]
            for q in nzrange(Rt, i)
                c = colour[Rt.rowval[q]]
                c > 0 && (forbidden[c] = j)
            end
        end
        c = 1
        while forbidden[c] == j
            c += 1
        end
        colour[j] = c
    end
    return colour
end

"""
    make_sparse_jacobian(sys, u_t0, u_dt, nu; correction=nothing, chunk=12)

The pattern (a sparse matrix of zeros, the `jac_prototype`) and an in-place `jac!(J, x, p, t)` filling it from
the frozen right-hand side by coloured forward-mode passes, `chunk` colours a pass.
"""
function make_sparse_jacobian(sys::CookSystem, u_t0::Real, u_dt::Real, nu::Int; correction=nothing, chunk::Int=12)
    rhsf!, play = make_system_rhs(sys, u_t0, u_dt, nu; correction, freeze=true)
    ix = _SysIndex(sys, play)
    S = _grid_system_pattern(sys, ix)
    proto = SparseMatrixCSC(S.m, S.n, S.colptr, S.rowval, zeros(length(S.nzval)))
    colour = _colour_columns(S)
    ncol = maximum(colour)
    D = ForwardDiff.Dual{_JacTag,Float64,chunk}
    xd = Vector{D}(undef, S.n); dxd = Vector{D}(undef, S.n)
    bycolour = [Int[] for _ in 1:ncol]
    for (j, c) in enumerate(colour)
        push!(bycolour[c], j)
    end
    jac! = function (Jm, x, p, t)
        for c0 in 1:chunk:ncol
            for j in eachindex(x)
                xd[j] = D(x[j], ForwardDiff.Partials(ntuple(_ -> 0.0, chunk)))
            end
            for c in c0:min(c0 + chunk - 1, ncol), j in bycolour[c]
                xd[j] = D(x[j], ForwardDiff.Partials(ntuple(q -> q == c - c0 + 1 ? 1.0 : 0.0, chunk)))
            end
            fill!(dxd, zero(D))
            rhsf!(dxd, xd, p, t)
            for c in c0:min(c0 + chunk - 1, ncol), j in bycolour[c], k in nzrange(Jm, j)
                Jm.nzval[k] = ForwardDiff.partials(dxd[Jm.rowval[k]], c - c0 + 1)
            end
        end
        return nothing
    end
    return proto, jac!, ncol
end

# whether a solve takes the sparse path: `:auto` for a Float64 solve of grid pieces only past `nmin` nodes in all
_use_sparse(sys::CookSystem, Tel, jacobian::Symbol; nmin=800) =
    jacobian === :sparse ? true : jacobian === :dense ? false :
    (Tel === Float64 && all(pc -> pc isa GridPiece, sys.pieces) && sum(pc -> pc.mesh.N, sys.pieces; init=0) >= nmin)
