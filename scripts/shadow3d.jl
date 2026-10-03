# The 3-D shadow of a plan: the planned cook re-simulated on a trimmed brisket flat (`BrisketFlat3D`) in the pit,
# through the cook system (`system_solve`), to serve. Loaded into a Julia session that has `using BrisketDynamics`
# (scripts/run_shadow3d.py, plot_shadow3d.py and render_shadow3d.py include it into the slab2d Tesseract's session).
#
# The cut: a flat tapering in thickness and width from its thick (point) end to its thin end, a fat cap on top, the
# dimensions `BRISKET_FLAT_TYPICAL` until the cut is measured. Half of it is solved, across its width (the centre
# plane a symmetry plane); the piece's area multiplier 2 makes the pit see the whole cut (heat, vapor, mass). The
# core node is the deepest point, the grid minimum the true cold spot.
#
# Or a reconstructed cut (tools/cutvision's labels, `shape_labels` in column-major order with `shape_dims` and cells of
# `shape_h` [m]): a `VoxelPiece` of the `meat`, its cells `coarsen` times the reconstruction's, the whole cut solved
# (area multiplier 1), the sparse Jacobian.

"""
    shadow3d(θ, φ, u, u_tknots, tsave; nl, nw, nt, dims, shape_labels, shape_dims, shape_h, meat, coarsen, t_wrap,
             t_pull, t_add, t_spritz, dt_spritz, m_spritz, t_salt, T0, Tpit0, T_amb, m0, rest_t, rest_h, rest_eps, rest_eta)

The planned cook to serve on the flat (`dims` its dimensions, `BRISKET_FLAT_TYPICAL`'s fields) or, given
`shape_labels`, on the reconstructed cut. Events at 1e8 s or later never happen. Returns the piece's outputs
(`system_outputs(...).piece[:brisket]`), the pit series, the mesh, each node's material (1 lean, 2 fat cap), its
bone weight, the dimensions (the flat's) and, for a reconstructed cut, each exposed face's outward direction.
"""
function shadow3d(θ, φ, u, u_tknots, tsave; nl::Int=12, nw::Int=6, nt::Int=8, dims=BRISKET_FLAT_TYPICAL,
                  shape_labels=nothing, shape_dims=nothing, shape_h::Real=0.002, meat::Symbol=:brisket, coarsen::Int=4,
                  t_wrap::Real=1e9, t_pull::Real=1e9, t_add::Real=1e9,
                  t_spritz::Real=1e9, dt_spritz::Real=3600.0, m_spritz::Real=0.05, t_salt::Real=0.0,
                  T0::Real=281.15, Tpit0::Real=293.15, T_amb::Real=293.15, m0::Real=7.0,
                  rest_t::Real=293.15, rest_h::Real=1.5, rest_eps::Real=0.10, rest_eta::Real=0.9)
    cfg = MeatConfig(; T0=T0, T_rest=rest_t, h_rest=rest_h, ε_rest=rest_eps, η_rest=rest_eta)
    pit = PitConfig(; T_amb=T_amb, Tpit0=Tpit0, m0=m0)
    faces = nothing
    if shape_labels === nothing
        pc = BrisketFlat3D(:brisket; length=dims.length, width_thick=dims.width_thick, width_thin=dims.width_thin,
                           thickness_thick=dims.thickness_thick, thickness_thin=dims.thickness_thin, cap=dims.cap,
                           nl, nw, nt, cfg)
        mult = 2.0                                                    # half the flat solved
    else
        L = reshape(Int8.(collect(shape_labels)), Tuple(Int.(collect(shape_dims))))
        lean = with_config(meat_preset(meat).cfg; T0, T_rest=rest_t, h_rest=rest_h, ε_rest=rest_eps, η_rest=rest_eta)
        pc = VoxelPiece(:brisket, L, shape_h; meat, coarsen, cfg=lean)
        faces = voxel_faces(L, shape_h; coarsen)
        mult = 1.0
    end
    sys = CookSystem(pc; pit)
    never(t) = !(t < 1e8)
    ev = CookEvent[]
    never(t_wrap) || push!(ev, CookEvent(:wrap, :brisket, t_wrap))
    never(t_pull) || push!(ev, CookEvent(:pull, :brisket, t_pull))
    never(t_spritz) || push!(ev, CookEvent(:spritz, :brisket, t_spritz; dt=dt_spritz, m=m_spritz))
    never(t_add) || push!(ev, CookEvent(:add, :pit, t_add))
    o = system_solve(sys, [collect(float.(θ))], [mult], collect(float.(φ)), collect(float.(u)),
                     collect(float.(u_tknots)), collect(float.(tsave)); events=ev, t_salt=float(t_salt))
    return (; piece=o.piece[:brisket], o.Tpit, o.m_fuel, o.pv_pit, mesh=pc.mesh, mat=pc.mat, bone=pc.bone, geometry=dims,
            faces=faces === nothing ? Int[] : reduce(vcat, ([d[1] d[2] d[3]] for d in faces)))
end
