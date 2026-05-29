for I in sectorlist
    println("------------------------------------")
    println("Fusion Trees $I")
    println("------------------------------------")
    ti = time()
    N = 5
    out = ntuple(n -> randsector(I), N)
    isdual = ntuple(n -> rand(Bool), N)
    in = rand(collect(⊗(out...)))
    numtrees = count(n -> true, fusiontrees(out, in, isdual))
    while !(0 < numtrees < 30)
        out = ntuple(n -> randsector(I), N)
        in = rand(collect(⊗(out...)))
        numtrees = count(n -> true, fusiontrees(out, in, isdual))
    end
    it = @constinferred fusiontrees(out, in, isdual)
    @constinferred Nothing iterate(it)
    f = @constinferred first(it)

    @testset "Fusion tree $I: printing" begin
        @test eval(Meta.parse(sprint(show, f))) == f
    end

    @timedtestset "Fusion tree $I: braiding" begin
        src = FusionTreeBlock{I}((out, ()), (isdual, ()))
        for i in 1:(N - 1)
            dst, U = @constinferred TK.artin_braid(src, i)
            dst2, U2 = @constinferred TK.artin_braid(dst, i; inv = true)
            @test src == dst2
            @test _isone(U2 * U)
        end
    end

    @timedtestset "Fusion tree $I: insertat" begin
        N = 3
        out2 = ntuple(n -> randsector(I), N)
        in2 = rand(collect(⊗(out2...)))
        isdual2 = ntuple(n -> rand(Bool), N)
        f2 = rand(collect(fusiontrees(out2, in2, isdual2)))
        for i in 1:N
            out1 = ntuple(n -> randsector(I), N)
            out1 = Base.setindex(out1, in2, i)
            in1 = rand(collect(⊗(out1...)))
            isdual1 = ntuple(n -> rand(Bool), N)
            isdual1 = Base.setindex(isdual1, false, i)
            f1 = rand(collect(fusiontrees(out1, in1, isdual1)))

            trees = @constinferred TK.insertat(f1, i, f2)
            @test norm(values(trees)) ≈ 1

            f1a, f1b = @constinferred TK.split(f1, $i)
            @test length(TK.insertat(f1b, 1, f1a)) == 1
            @test first(TK.insertat(f1b, 1, f1a)) == (f1 => 1)

            Af1 = convert(SparseArray, f1)
            Af2 = convert(SparseArray, f2)
            Af = TensorOperations.tensorcontract(
                1:(2N),
                Af1, [1:(i - 1); -1; N - 1 .+ ((i + 1):(N + 1))],
                Af2, [i - 1 .+ (1:N); -1]
            )
            Af′ = zero(Af)
            for (f, coeff) in trees
                Af′ .+= coeff .* convert(SparseArray, f)
            end
            @test Af ≈ Af′
        end
    end

    @timedtestset "Fusion tree $I: merging" begin
        N = 3
        out1 = ntuple(n -> randsector(I), N)
        in1 = rand(collect(⊗(out1...)))
        f1 = rand(collect(fusiontrees(out1, in1)))
        out2 = ntuple(n -> randsector(I), N)
        in2 = rand(collect(⊗(out2...)))
        f2 = rand(collect(fusiontrees(out2, in2)))

        @constinferred TK.merge(f1, f2, first(in1 ⊗ in2), 1)
        @test dim(in1) * dim(in2) ≈ sum(
            abs2(coeff) * dim(c) for c in in1 ⊗ in2
                for μ in 1:Nsymbol(in1, in2, c)
                for (f, coeff) in TK.merge(f1, f2, c, μ)
        )

        for c in in1 ⊗ in2
            for μ in 1:Nsymbol(in1, in2, c)
                trees1 = TK.merge(f1, f2, c, μ)

                Af1 = convert(SparseArray, f1)
                Af2 = convert(SparseArray, f2)
                Af0 = convert(
                    SparseArray,
                    FusionTree((f1.coupled, f2.coupled), c, (false, false), (), (μ,))
                )
                _Af = TensorOperations.tensorcontract(
                    1:(N + 2), Af1, [1:N; -1], Af0, [-1; N + 1; N + 2]
                )
                Af = TensorOperations.tensorcontract(
                    1:(2N + 1), Af2, [N .+ (1:N); -1], _Af, [1:N; -1; 2N + 1]
                )
                Af′ = zero(Af)
                for (f, coeff) in trees1
                    Af′ .+= coeff .* convert(SparseArray, f)
                end
                @test Af ≈ Af′
            end
        end
    end

    N = 3
    out = ntuple(n -> randsector(I), N)
    numtrees = count(n -> true, fusiontrees((out..., map(dual, out)...)))
    while !(0 < numtrees < 100)
        out = ntuple(n -> randsector(I), N)
        numtrees = count(n -> true, fusiontrees((out..., map(dual, out)...)))
    end
    incoming = rand(collect(⊗(out...)))
    isdual1 = ntuple(n -> rand(Bool), N)
    isdual2 = ntuple(n -> rand(Bool), N)
    src = FusionTreeBlock{I}((out, out), (isdual1, isdual2))
    A = map(fusiontensor, fusiontrees(src))

    @timedtestset "Double fusion tree $I: repartitioning" begin
        for n in 0:(2 * N)
            dst, U = @constinferred TK.repartition(src, $n)

            dst′, U′ = TK.repartition(dst, N)
            @test _isone(U′ * U)

            all_inds = (
                ntuple(identity, N)...,
                reverse(ntuple(i -> i + N, N))...,
            )
            p₁ = ntuple(i -> all_inds[i], n)
            p₂ = reverse(ntuple(i -> all_inds[i + n], 2N - n))
            A′ = map(Base.Fix2(permutedims, (p₁..., p₂...)), A)
            A″ = map(fusiontensor, fusiontrees(dst))
            for (i, Ai) in enumerate(A′)
                @test Ai ≈ sum(A″ .* U[:, i])
            end
        end
    end

    @timedtestset "Double fusion tree $I: permutation" begin
        if BraidingStyle(I) isa SymmetricBraiding
            for n in 0:(2N)
                p = (randperm(2 * N)...,)
                p1, p2 = p[1:n], p[(n + 1):(2N)]
                ip = invperm(p)
                ip1, ip2 = ip[1:N], ip[(N + 1):(2N)]

                dst, U = @constinferred TensorKit.permute(src, (p1, p2))

                dst′, U′ = @constinferred TensorKit.permute(dst, (ip1, ip2))
                @test _isone(U′ * U)

                A′ = map(Base.Fix2(permutedims, (p1..., p2...)), A)
                A″ = map(fusiontensor, fusiontrees(dst))
                for (i, Ai) in enumerate(A′)
                    @test Ai ≈ sum(A″ .* U[:, i])
                end
            end
        end
    end
    tf = time()
    printstyled(
        "Finished fusion tree $I tests in ",
        string(round(tf - ti; sigdigits = 3)),
        " seconds."; bold = true, color = Base.info_color()
    )
    println()
end
