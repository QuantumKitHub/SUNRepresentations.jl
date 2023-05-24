for I in sectorlist
    println("------------------------------------")
    println("Sector $I")
    println("------------------------------------")
    ti = time()
    @testset "Sector $I: Basic properties" begin
        s = (randsector(I), randsector(I), randsector(I))
        @test eval(Meta.parse(sprint(show,s[1]))) == s[1]
        @test @constinferred(hash(s[1])) == hash(deepcopy(s[1]))
        @test @constinferred(one(s[1])) == @constinferred(one(I))
        @constinferred dual(s[1])
        @constinferred dim(s[1])
        @constinferred frobeniusschur(s[1])
        @constinferred Nsymbol(s...)
        @constinferred Rsymbol(s...)
        @constinferred Bsymbol(s...)
        @constinferred Fsymbol(s..., s...)
        it = @constinferred s[1] ⊗ s[2]
        @constinferred ⊗(s..., s...)
        for i = 1:3
            @test 1 == @constinferred Nsymbol(s[i], conj(s[i]), one(s[i]))
        end
    end
    @timedtestset "Sector $I: Value iterator" begin
        @test eltype(values(I)) == I
        sprev = one(I)
        for (i, s) in enumerate(values(I))
            @test !isless(s, sprev) # confirm compatibility with sort order
            if Base.IteratorSize(values(I)) == Base.IsInfinite() && I <: ProductSector
                @test_throws ArgumentError values(I)[i]
                @test_throws ArgumentError TensorKit.findindex(values(I), s)
            elseif hasmethod(Base.getindex, Tuple{typeof(values(I)), Int})
                @test s == @constinferred (values(I)[i])
                @test TensorKit.findindex(values(I), s) == i
            end
            sprev = s
            i >= 10 && break
        end
        @test one(I) == first(values(I))
        if Base.IteratorSize(values(I)) == Base.IsInfinite() && I <: ProductSector
            @test_throws ArgumentError TensorKit.findindex(values(I), one(I))
        elseif hasmethod(Base.getindex, Tuple{typeof(values(I)), Int})
            @test (@constinferred TensorKit.findindex(values(I), one(I))) == 1
            for s in smallset(I)
                @test (@constinferred values(I)[TensorKit.findindex(values(I), s)]) == s
            end
        end
    end
    @timedtestset "Sector $I: fusion tensor and F-move and R-move" begin
        for a in smallset(I), b in smallset(I)
            for c in ⊗(a,b)
                X1 = permutedims(fusiontensor(a, b, c), (2, 1, 3, 4))
                X2 = fusiontensor(b, a, c)
                l =dim(a)*dim(b)*dim(c)
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
                    @tensor f1[-1,-2,-3,-4] := conj(Y2[a,f,d,-4])*conj(Y1[b,c,f,-3])*
                                                X1[a,b,e,-1] * X2[e,c,d,-2]
                    f2 = Fsymbol(a,b,c,d,e,f)*dim(d)
                    @test isapprox(f1, f2; atol = 1000*eps(), rtol = 1000*eps())
                end
            end
        end
    end
    @timedtestset "Sector $I: Unitarity of F-move" begin
        for a in smallset(I), b in smallset(I), c in smallset(I)
            for d in ⊗(a,b,c)
                es = collect(intersect(⊗(a,b), map(dual, ⊗(c,dual(d)))))
                fs = collect(intersect(⊗(b,c), map(dual, ⊗(dual(d),a))))
                Fblocks = Vector{Any}()
                for e in es
                    for f in fs
                        Fs = Fsymbol(a,b,c,d,e,f)
                        push!(Fblocks, reshape(Fs, (size(Fs, 1)*size(Fs, 2), size(Fs, 3)*size(Fs, 4))))
                    end
                end
                F = hvcat(length(fs), Fblocks...)
                @test F'*F ≈ one(F)
            end
        end
    end
    @testset "Sector $I: Pentagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I), d in smallset(I)
            @test pentagon_equation(a, b, c, d; atol=1e-12, rtol=1e-12)
        end
    end
    @testset "Sector $I: Hexagon equation" begin
        for a in smallset(I), b in smallset(I), c in smallset(I)
            @test hexagon_equation(a, b, c; atol=1e-12, rtol=1e-12)
        end
    end
    tf = time()
    printstyled("Finished sector $I tests in ",
                string(round(tf-ti; sigdigits=3)),
                " seconds."; bold = true, color = Base.info_color())
    println()
end
