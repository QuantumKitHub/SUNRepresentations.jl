println("------------------------------------")
println("Generic SUNIrrep tests")
println("------------------------------------")

using SUNRepresentations: cartanmatrix, inverse_cartanmatrix, dimname, dynkin_label
using Latexify: latexify, @L_str

@timedtestset "Basic tests for SUNIrrep{$N}:" for N in 2:5
    I1 = SUNIrrep(tuple(sort(rand(1:9, N); rev = true)..., 1))
    I2 = SUNIrrep(tuple(sort(rand(1:9, N); rev = true)..., 1))
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
    for i in 1:(length(v) - 1)
        @test isless(v[i], v[i + 1])
    end
    @test v[end] == highest_weight(I1)
    s = sprint(show, v[1])
    for k in 1:N
        @test parse(Int, s[17 + 2 * k]) == weight(I1)[k]
    end

    @test inv(cartanmatrix(I1)) ≈ inverse_cartanmatrix(I1)
    @test SUNIrrep{N}("1") === one(SUNIrrep{N})
end

@timedtestset "Names of SU3Irrep:" begin
    dimnames = [
        "1", "3⁺", "3", "6⁺", "8", "6", "10", "15⁺", "15", "10⁺", "15′⁺", "24",
        "27", "24⁺", "15′", "21", "35", "42⁺", "42", "35⁺",
    ]
    for (i, I) in enumerate(Iterators.take(values(SU3Irrep), length(dimnames)))
        @test dimname(I) == dimnames[i]
        @test SU3Irrep(dimnames[i]) === I
        @test SU3Irrep(collect(dynkin_label(I))) === I
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
        @test SU4Irrep(collect(dynkin_label(I))) === I
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
        @test SU5Irrep(collect(dynkin_label(I))) === I
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
