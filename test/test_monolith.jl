using BrisketDynamics: monolith_solve, monolith_jacobian, picard_cook, Humidity

const MONO_T_F = 8 * 3600.0
const MONO_TSAVE = collect(0.0:600.0:MONO_T_F)
const MONO_UTK = collect(range(0.0, MONO_T_F; length=8))
const MONO_U = fill(0.5, 8)
const MONO_PIT = PitConfig(; T_amb=300.0)

@testset "monolith: one coupled solve vs the Picard exchange" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    mono = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    @test length(mono.Tpit) == length(MONO_TSAVE)
    @test all(isfinite, mono.Tcore) && all(isfinite, mono.Tpit)
    @test mono.Tcore[end] > mono.Tcore[1] + 30
    @test mono.m_fuel[end] < mono.m_fuel[1]
    # the meat adds vapor while evaporating and removes it while condensing
    @test all((mono.pv_pit .- mono.pv_base) .* sign.(mono.j_evap) .>= -1e-9)

    # Picard converges TOWARD the monolith as K grows; the residual left at
    # K=8 is the exchange-grid interpolation, not the coupling truncation
    d = Dict{Int,NTuple{2,Float64}}()
    for K in (1, 2, 4, 8)
        pc = picard_cook(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; K, pit=MONO_PIT)
        d[K] = (
            maximum(abs.(pc.pit.Tpit .- mono.Tpit)),
            maximum(abs.(pc.meat.Tcore .- mono.Tcore)),
        )
    end
    @info "monolith vs Picard(K): max|ΔT_pit|, max|ΔT_core| [K]" d[1] d[2] d[4] d[8]
    @test d[8][1] < d[2][1]
    @test d[8][2] < d[2][2]
    @test d[8][1] < 2.0
    @test d[8][2] < 2.0

    # cost: one coupled solve vs the two-pass exchange (both warm)
    t_mono = @elapsed monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    t_pic = @elapsed picard_cook(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; K=2, pit=MONO_PIT)
    @info "wall: monolith $(round(t_mono; digits=3)) s, Picard K=2 $(round(t_pic; digits=3)) s"
end

@testset "monolith: forward-mode Jacobian vs finite differences" begin
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    f = o -> vcat(o.Tpit, o.Tcore)
    t_j = @elapsed J = monolith_jacobian(f, θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT)
    @test size(J) == (2 * length(MONO_TSAVE), 14)
    @test all(isfinite, J)
    @info "wall: 14-column forward Jacobian $(round(t_j; digits=2)) s (incl. compile)"
    x = vcat(θ, φ)
    for (j, name) in ((1, "D_eff"), (3, "h_c"), (9, "UA"), (12, "u_leak"))
        h = 1e-3 * abs(x[j])
        xp, xm = copy(x), copy(x)
        xp[j] += h
        xm[j] -= h
        fp = f(monolith_solve(xp[1:7], xp[8:14], MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT))
        fm = f(monolith_solve(xm[1:7], xm[8:14], MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT))
        fd = (fp .- fm) ./ (2h)
        err = maximum(abs.(J[:, j] .- fd)) / (maximum(abs.(fd)) + 1e-12)
        @info "FD gate $name: relative error $(round(err; sigdigits=3))"
        @test err < 2e-2
    end
end

@testset "monolith: stacked Jacobian and JVP (the Tesseract's derivative endpoints)" begin
    using BrisketDynamics: monolith_stack, monolith_stack_jacobian, monolith_stack_jvp
    θ, φ = MEAT_θ_DEFAULT, PIT_φ_GRAD
    nu = length(MONO_U)
    tw, tp = 3 * 3600.0, 7 * 3600.0
    sel = [1, 9, 15, 14 + nu + 1, 14 + nu + 2]  # D_eff, UA, u_knots[1], t_wrap, t_pull
    J = monolith_stack_jacobian(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE, tw, tp, sel; pit=MONO_PIT)
    o = monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_wrap=tw, t_pull=tp)
    @test size(J) == (length(monolith_stack(o)), length(sel))
    @test all(isfinite, J)
    dv = zeros(14 + nu + 2)
    dv[sel] .= [1e-9, 2.0, 0.1, 600.0, 600.0]
    d = monolith_stack_jvp(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE, tw, tp, dv; pit=MONO_PIT)
    ref = J * dv[sel]
    # a 1-partial and a 5-partial Dual solve take slightly different adaptive
    # steps (the error norm includes the partials); the pit / heat-load rows
    # differ by ~1e-3 of the largest entry
    @test maximum(abs.(d .- ref)) / (maximum(abs.(ref)) + 1e-12) < 5e-3
    h = 60.0
    fp = monolith_stack(monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_wrap=tw, t_pull=tp + h))
    fm = monolith_stack(monolith_solve(θ, φ, MONO_U, MONO_UTK, MONO_TSAVE; pit=MONO_PIT, t_wrap=tw, t_pull=tp - h))
    fd = (fp .- fm) ./ (2h)
    @test maximum(abs.(J[:, 5] .- fd)) / (maximum(abs.(fd)) + 1e-12) < 2e-2
    @test abs(J[end - 1, 5]) > 0  # juiciness at serve moves with the pull
end
