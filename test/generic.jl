println("------------------------------------")
println("Generic SUNIrrep tests")
println("------------------------------------")

using SUNRepresentations: cartanmatrix, inverse_cartanmatrix, dimname, dynkin_label
using Latexify: latexify, @L_str
using SUNRepresentations.TensorKitSectors: ⊗, Nsymbol, dual

@timedtestset "Basic tests for SUNIrrep{$N}:" for N in 2:5
    I1 = SUNIrrep{N}(sort!(rand(1:9, N - 1); rev = true)..., 0)
    I2 = SUNIrrep{N}(sort!(rand(1:9, N - 1); rev = true)..., 0)
    @constinferred dim(I1)
    d = 0
    for (I, NI) in @constinferred directproduct(I1, I2)
        d += NI * dim(I)
    end
    @test d == dim(I1) * dim(I2)
    if dim(I2) < dim(I1)
        I1, I2 = I2, I1
    end
    b = @constinferred basis(I1)
    (v, s) = @constinferred Nothing iterate(b)
    @constinferred Nothing iterate(b, s)
    v = @constinferred collect(b)
    @test @constinferred(length(b)) == length(v)
    for i in 1:(length(v) - 1)
        @test isless(v[i], v[i + 1])
    end
    @test v[end] == highest_weight(I1)
    s = sprint(show, v[1])
    for k in 1:N
        @test parse(Int, s[17 + 2 * k]) == weight(I1)[k]
    end

    @test I1.a == @constinferred dynkin_label(I1)
    @test I1.I == @constinferred weight(I1)

    @test inv(cartanmatrix(I1)) ≈ inverse_cartanmatrix(I1)
    @test SUNIrrep{N}("1") === one(SUNIrrep{N})
end

@timedtestset "Direct product of SUNIrrep{$N}:" for N in 2:6
    irreps = collect(Iterators.take(values(SUNIrrep{N}), 20))
    @constinferred collect(irreps[end] ⊗ irreps[end - 1])
    @constinferred Nsymbol(irreps[end], irreps[end - 1], irreps[end - 2])
    for a in irreps, b in irreps
        # deterministic order, independent of `hash` (#54)
        ab = collect(a ⊗ b)
        @test issorted(ab) && allunique(ab)
        dp = directproduct(a, b)
        @test issetequal(keys(dp), ab)
        @test ab == collect(b ⊗ a)
        @test sum(c -> dp[c] * dim(c), ab) == dim(a) * dim(b)
        for c in irreps
            n = Nsymbol(a, b, c)
            @test n == get(dp, c, 0)
            @test n == Nsymbol(b, a, c) == Nsymbol(a, dual(c), dual(b)) ==
                Nsymbol(dual(a), dual(b), dual(c))
        end
    end
end

@timedtestset "Large direct products" begin
    for (a, b) in (
            (SU3Irrep(9, 9), SU3Irrep(8, 10)),
            (SU4Irrep(3, 2, 3), SU4Irrep(2, 3, 2)),
            (SU5Irrep(2, 1, 1, 2), SU5Irrep(1, 2, 2, 1)),
        )
        ab = collect(a ⊗ b)
        @test issorted(ab) && allunique(ab)
        dp = directproduct(a, b)
        @test issetequal(keys(dp), ab)
        @test sum(c -> dp[c] * dim(c), ab) == dim(a) * dim(b)
        @test all(c -> Nsymbol(a, b, c) == dp[c] == Nsymbol(dual(a), dual(b), dual(c)), ab)
    end
end

@timedtestset "Illegal constructors" begin
    @test_throws ArgumentError SUNIrrep((1, 2))
    @test_throws ArgumentError SUNIrrep(1, 2)
    @test_throws TypeError SUNIrrep{2, 2}((1, 2))
end

@timedtestset "Names of SU3Irrep:" begin
    dimnames = [
        "1", "3⁺", "3", "6⁺", "8", "6", "10", "15⁺", "15", "10⁺", "15′⁺", "24",
        "27", "24⁺", "15′", "21", "35", "42⁺", "42", "35⁺",
    ]
    for (i, I) in enumerate(Iterators.take(values(SU3Irrep), length(dimnames)))
        @test dimname(I) == dimnames[i]
        @test SU3Irrep(dimnames[i]) === I
        @test SU3Irrep(dynkin_label(I)) === I
    end
end

@timedtestset "Names of SU4Irrep:" begin
    dimnames = [
        "1", "4⁺", "6", "4", "10", "20", "20′", "15", "20⁺", "10⁺", "20″", "45",
        "60⁺", "50", "36⁺", "64", "60", "36", "45⁺", "20″⁺",
    ]
    for (i, I) in enumerate(Iterators.take(values(SU4Irrep), length(dimnames)))
        @test dimname(I) == dimnames[i]
        @test SU4Irrep(dimnames[i]) === I
        @test SU4Irrep(dynkin_label(I)) === I
    end
end

@timedtestset "Names of SU5Irrep:" begin
    dimnames = [
        "1", "5⁺", "10⁺", "10", "5", "15⁺", "40", "50", "45", "75", "50⁺", "24",
        "45⁺", "40⁺", "15", "35", "105", "175′", "175″⁺", "126",
    ]
    for (i, I) in enumerate(Iterators.take(values(SU5Irrep), length(dimnames)))
        @test dimname(I) == dimnames[i]
        @test SU5Irrep(dimnames[i]) === I
        @test SU5Irrep(dynkin_label(I)) === I
    end
end

@timedtestset "LaTeX names of SU3Irrep" begin
    latexnames = [
        L"$\textbf{1}$", L"$\overline{\textbf{3}}$", L"$\textbf{3}$",
        L"$\overline{\textbf{6}}$", L"$\textbf{8}$", L"$\textbf{6}$",
        L"$\textbf{10}$", L"$\overline{\textbf{15}}$", L"$\textbf{15}$",
        L"$\overline{\textbf{10}}$", L"$\overline{\textbf{15}}^\prime$",
        L"$\textbf{24}$", L"$\textbf{27}$", L"$\overline{\textbf{24}}$",
        L"$\textbf{15}^\prime$", L"$\textbf{21}$", L"$\textbf{35}$",
        L"$\overline{\textbf{42}}$", L"$\textbf{42}$",
        L"$\overline{\textbf{35}}$",
    ]
    for (i, I) in enumerate(Iterators.take(values(SU3Irrep), length(latexnames)))
        @test latexify(I) == latexnames[i]
    end
end

@timedtestset "LaTeX names of SU4Irrep" begin
    latexnames = [
        L"$\textbf{1}$", L"$\overline{\textbf{4}}$", L"$\textbf{6}$",
        L"$\textbf{4}$", L"$\textbf{10}$", L"$\textbf{20}$",
        L"$\textbf{20}^\prime$", L"$\textbf{15}$",
        L"$\overline{\textbf{20}}$", L"$\overline{\textbf{10}}$",
        L"$\textbf{20}^{\prime\prime}$", L"$\textbf{45}$",
        L"$\overline{\textbf{60}}$", L"$\textbf{50}$",
        L"$\overline{\textbf{36}}$", L"$\textbf{64}$", L"$\textbf{60}$",
        L"$\textbf{36}$", L"$\overline{\textbf{45}}$",
        L"$\overline{\textbf{20}}^{\prime\prime}$",
    ]
    for (i, I) in enumerate(Iterators.take(values(SU4Irrep), length(latexnames)))
        @test latexify(I) == latexnames[i]
    end
end

@timedtestset "LaTeX names of SU5Irrep" begin
    latexnames = [
        L"$\textbf{1}$", L"$\overline{\textbf{5}}$",
        L"$\overline{\textbf{10}}$", L"$\textbf{10}$", L"$\textbf{5}$",
        L"$\overline{\textbf{15}}$", L"$\textbf{40}$", L"$\textbf{50}$",
        L"$\textbf{45}$", L"$\textbf{75}$", L"$\overline{\textbf{50}}$",
        L"$\textbf{24}$", L"$\overline{\textbf{45}}$",
        L"$\overline{\textbf{40}}$", L"$\textbf{15}$", L"$\textbf{35}$",
        L"$\textbf{105}$", L"$\textbf{175}^\prime$",
        L"$\overline{\textbf{175}}^{\prime\prime}$", L"$\textbf{126}$",
    ]
    for (i, I) in enumerate(Iterators.take(values(SU5Irrep), length(latexnames)))
        i > length(latexnames) && break
        @test latexify(I) == latexnames[i]
    end
end

@timedtestset "Dimension of SUNIrrep{$N}" for N in 2:12
    # against the Weyl dimension formula in rational arithmetic
    for _ in 1:10
        I = SUNIrrep{N}(sort!(rand(0:9, N - 1); rev = true)..., 0)
        w = weight(I)
        @test dim(I) == prod((k2 - k1 + w[k1] - w[k2]) // (k2 - k1) for k2 in 2:N for k1 in 1:(k2 - 1))
    end
    @test dim(one(SUNIrrep{N})) == 1
end

@timedtestset "Dimension of SUNIrrep{$N} from its basis" for N in 2:6
    for s in Iterators.take(values(SUNIrrep{N}), 50)
        @test dim(s) == count(Returns(true), basis(s))
    end
end

@timedtestset "Large dimensions of SUNIrrep{$N}" for N in 2:7
    # all Dynkin labels equal to a: dim = (a + 1)^(N(N - 1)/2)
    for a in (1, 3, 10, 64, 255)
        d = big(a + 1)^(N * (N - 1) ÷ 2)
        s = SUNIrrep{N}(ntuple(Returns(a), N - 1))
        if d <= typemax(Int)
            @test dim(s) == d
        else
            @test_throws OverflowError dim(s)
        end
    end
end

@timedtestset "Dimensions of few-box irreps of SUNIrrep{$N}" for N in (6, 10, 22, 40)
    e(k) = ntuple(i -> Int(i == k), N - 1)
    I = SUNIrrep{N}
    for (s, d) in [
            (I(e(1)), N), (I(e(1) .+ e(N - 1)), N^2 - 1),
            (I(e(2)), binomial(N, 2)), (I(e(3)), binomial(N, 3)),
            (I(3 .* e(1)), binomial(N + 2, 3)), (I(e(1) .+ e(2)), N * (N^2 - 1) ÷ 3),
        ]
        @test dim(s) == dim(dual(s)) == d
    end
end

@timedtestset "Dimension with large intermediate products" begin
    # fits in an Int, while the products of the row-by-row Weyl formula overflow an Int128
    s = SUNIrrep{22}(ntuple(i -> 3 * (i == 4) + 2 * (i == 16), 21))
    w = weight(s)
    @test dim(s) == prod(big(k2 - k1 + w[k1] - w[k2]) // (k2 - k1) for k2 in 2:22 for k1 in 1:(k2 - 1))
end

@timedtestset "Dimension of symmetric powers of SUNIrrep{$N} (BigInt Weyl formula)" for N in 10:12
    s = SUNIrrep{N}(255, ntuple(Returns(0), N - 2)...)
    d = binomial(big(254 + N), N - 1)
    if d <= typemax(Int)
        @test dim(s) == d
    else
        @test_throws OverflowError dim(s)
    end
end
