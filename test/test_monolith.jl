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
    @test all(mono.pv_pit .>= mono.pv_base)  # the meat only adds vapor

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
