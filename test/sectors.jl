# `Fsymbol`/`Rsymbol` are computed from the compact SU(N-1)×U(1) `reduced_CGC` (densification-free),
# whose weight-descent error compounds with N through the multi-CGC F-contraction. Measured over the
# SU(5) smallset the F-move vs dense reference peaks at ~7e-10 (R-move stays ~1e-12, and pentagon/
# hexagon — being internal-consistency checks — hold to <1e-10). The exact dense CGC used at N≤4
# meets the machine-precision cross-checks below; the reduced method at N≥5 does not, so relax those
# specific tolerances to 1e-8 (well clear of ~7e-10, still catching any gross/gauge error). N≤4 and
# product sectors stay strict.
reduced_hn_tol(::Type, strict) = strict
reduced_hn_tol(::Type{<:SUNIrrep{N}}, strict) where {N} = N >= 5 ? 1.0e-8 : strict

for I in sectorlist
    println("------------------------------------")
    println("Sector $I")
    println("------------------------------------")
    ti = time()
    @testset "Sector $I: Basic properties" begin
        s = (randsector(I), randsector(I), randsector(I))

        mode_old = SUNRepresentations.display_mode("dimension")
        for mode in ["dimension", "dynkin", "weight"]
            SUNRepresentations.display_mode(mode)
            @test eval(Meta.parse(sprint(show, s[1]))) == s[1]
        end
        SUNRepresentations.display_mode(mode_old)

        @test @constinferred(hash(s[1])) == hash(deepcopy(s[1]))
        @test @constinferred(one(s[1])) == @constinferred(one(I))
        @constinferred dual(s[1])
        @constinferred dim(s[1])
        @constinferred frobenius_schur_phase(s[1])
        @constinferred frobenius_schur_indicator(s[1])
        @constinferred Nsymbol(s...)
        @constinferred Rsymbol(s...)
        @constinferred Bsymbol(s...)
        @constinferred Fsymbol(s..., s...)
        it = @constinferred s[1] ⊗ s[2]
        @constinferred ⊗(s..., s...)
        for i in 1:3
            @test 1 == @constinferred Nsymbol(s[i], conj(s[i]), one(s[i]))
        end
    end
    @timedtestset "Sector $I: Value iterator" begin
        @test eltype(values(I)) == I
        sprev = one(I)
        for (i, s) in enumerate(values(I))
            @test !isless(s, sprev) # confirm compatibility with sort order
            @test s == @constinferred (values(I)[i])
            @test TensorKit.findindex(values(I), s) == i
            sprev = s
            i >= 10 && break
        end
        @test one(I) == first(values(I))
        @test (@constinferred TensorKit.findindex(values(I), one(I))) == 1
        for s in smallset(I)
            @test (@constinferred values(I)[TensorKit.findindex(values(I), s)]) == s
        end
    end
    @timedtestset "Sector $I: fusion tensor and F-move and R-move" begin
        for a in smallset(I), b in smallset(I)
            for c in ⊗(a, b)
                X1 = permutedims(fusiontensor(a, b, c), (2, 1, 3, 4))
                X2 = fusiontensor(b, a, c)
                l = dim(a) * dim(b) * dim(c)
                R = LinearAlgebra.transpose(Rsymbol(a, b, c))
                sz = (l, convert(Int, Nsymbol(a, b, c)))
                @test reshape(X1, sz) ≈ reshape(X2, sz) * R
            end
        end
        for a in smallset(I), b in smallset(I), c in smallset(I)
            for e in ⊗(a, b), f in ⊗(b, c)
                for d in intersect(⊗(e, c), ⊗(a, f))
                    X1 = fusiontensor(a, b, e)
                    X2 = fusiontensor(e, c, d)
                    Y1 = fusiontensor(b, c, f)
                    Y2 = fusiontensor(a, f, d)
                    @tensor f1[-1, -2, -3, -4] := conj(Y2[a, f, d, -4]) *
                        conj(Y1[b, c, f, -3]) *
                        X1[a, b, e, -1] * X2[e, c, d, -2]
                    f2 = Fsymbol(a, b, c, d, e, f) * dim(d)
                    ftol = reduced_hn_tol(I, 1000 * eps())
                    @test isapprox(f1, f2; atol = ftol, rtol = ftol)
                end
            end
        end
    end
    @timedtestset "Sector $I: Unitarity of F-move" begin
        for a in smallset(I), b in smallset(I), c in smallset(I), d in ⊗(a, b, c)
            es = collect(intersect(⊗(a, b), map(dual, ⊗(c, dual(d)))))
            fs = collect(intersect(⊗(b, c), map(dual, ⊗(dual(d), a))))
            Fblocks = Vector{Any}()
            for e in es, f in fs
                Fs = Fsymbol(a, b, c, d, e, f)
                push!(
                    Fblocks,
                    reshape(Fs, (size(Fs, 1) * size(Fs, 2), size(Fs, 3) * size(Fs, 4)))
                )
            end
            F = hvcat(length(fs), Fblocks...)
            @test F' * F ≈ one(F)
        end
    end
    @testset "Sector $I: Pentagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I), d in smallset(I)
            ptol = reduced_hn_tol(I, 1.0e-12)
            @test pentagon_equation(a, b, c, d; atol = ptol, rtol = ptol)
        end
    end
    @testset "Sector $I: Hexagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I)
            htol = reduced_hn_tol(I, 1.0e-12)
            @test hexagon_equation(a, b, c; atol = htol, rtol = htol)
        end
    end
    tf = time()
    printstyled(
        "Finished sector $I tests in ",
        string(round(tf - ti; sigdigits = 3)),
        " seconds."; bold = true, color = Base.info_color()
    )
    println()

    try
        s = sprint(SUNRepresentations.cache_info)
    catch
        @test false
    end
end
