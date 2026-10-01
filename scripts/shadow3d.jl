# The 3-D shadow of a plan: the planned cook re-simulated on a brick of the cut (`Brick3D`) in the pit, through the
# cook system (`system_solve`), to serve. Loaded into a Julia session that has `using BrisketDynamics`
# (scripts/run_shadow3d.py and scripts/plot_brick3d.py include it into the slab2d Tesseract's session).
#
# Geometry: a flat of full thickness 2L (the 1-D half-thickness doubled), a footprint W × Lz whose top face carries
# half the 1-D exposed area A_meat (A_meat counts both faces of a flat, as the 2-D shadow's radius does), at an
# aspect Lz/W. Only one eighth of the cut is solved: the brick [0, L] × [0, W/2] × [0, Lz/2] with its three faces
# at the origin as symmetry planes (insulated) and the top, one side and one end exposed; the piece's area
# multiplier 8 makes the pit see the whole cut (heat, vapor, mass all scale with it). The core node is the cut's
# centre, the grid minimum the true cold spot.

"""
    brick3d_geometry(L, A_meat; aspect=2.0) -> (; thickness, width, length)

The full cut's dimensions [m]: thickness 2L, top face A_meat/2 at length/width = `aspect`.
"""
function brick3d_geometry(L, A_meat; aspect=2.0)
    width = sqrt(A_meat / 2 / aspect)
    return (; thickness=2L, width, length=aspect * width)
end

"""
    shadow3d(θ, φ, u, u_tknots, tsave; nx, ny, nz, aspect, A_meat, t_wrap, t_pull, t_add, t_spritz, dt_spritz,
             m_spritz, t_salt, T0, Tpit0, T_amb, m0, rest_t, rest_h, rest_eps, rest_eta)

The planned cook on the eighth-brick to serve. Events at 1e8 s or later never happen. Returns the piece's outputs
(`system_outputs(...).piece[:brisket]`), the pit series, the mesh and the geometry.
"""
function shadow3d(θ, φ, u, u_tknots, tsave; nx::Int=6, ny::Int=8, nz::Int=12, aspect::Real=2.0,
                  A_meat::Real=0.12, t_wrap::Real=1e9, t_pull::Real=1e9, t_add::Real=1e9,
                  t_spritz::Real=1e9, dt_spritz::Real=3600.0, m_spritz::Real=0.05, t_salt::Real=0.0,
                  T0::Real=281.15, Tpit0::Real=293.15, T_amb::Real=293.15, m0::Real=7.0,
                  rest_t::Real=293.15, rest_h::Real=1.5, rest_eps::Real=0.10, rest_eta::Real=0.9)
    g = brick3d_geometry(θ[4], A_meat; aspect)
    cfg = MeatConfig(; T0=T0, T_rest=rest_t, h_rest=rest_h, ε_rest=rest_eps, η_rest=rest_eta)
    pit = PitConfig(; T_amb=T_amb, Tpit0=Tpit0, m0=m0)
    pc = Brick3D(:brisket; Lx=g.thickness / 2, Ly=g.width / 2, Lz=g.length / 2, nx, ny, nz,
                 exposed=(false, true, false, true, false, true), cfg)
    sys = CookSystem(pc; pit)
    never(t) = !(t < 1e8)
    ev = CookEvent[]
    never(t_wrap) || push!(ev, CookEvent(:wrap, :brisket, t_wrap))
    never(t_pull) || push!(ev, CookEvent(:pull, :brisket, t_pull))
    never(t_spritz) || push!(ev, CookEvent(:spritz, :brisket, t_spritz; dt=dt_spritz, m=m_spritz))
    never(t_add) || push!(ev, CookEvent(:add, :pit, t_add))
    o = system_solve(sys, [collect(float.(θ))], [8.0], collect(float.(φ)), collect(float.(u)),
                     collect(float.(u_tknots)), collect(float.(tsave)); events=ev, t_salt=float(t_salt))
    return (; piece=o.piece[:brisket], o.Tpit, o.m_fuel, o.pv_pit, mesh=pc.mesh, geometry=g)
end
