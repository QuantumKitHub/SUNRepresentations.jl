println("------------------------------------")
println("Generic SUNIrrep tests")
println("------------------------------------")
@timedtestset "Basic tests for SUNIrrep{$N}:" for N in 2:5
    I1 = SUNIrrep(tuple(sort(rand(1:9, N); rev=true)..., 1))
    I2 = SUNIrrep(tuple(sort(rand(1:9, N); rev=true)..., 1))
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
end

@timedtestset "Properties of SUNIrrep{2}:" begin
    indices = [1, 4, 10, 20, 35, 56, 84, 120, 165, 220]
    for i in 1:10
        I = SUNIrrep(i, 0)
        @test dynkin_label(I) == (i,)
        @test congruency(I) == i % 2
        @test index(I) == indices[i]
        @test dimname(I) == "$(dim(I))"
    end
end

@timedtestset "Properties of SUNIrrep{3}:" begin
    
    
    
    irreps = (SUNIrrep(1, 0, 0), SUNIrrep(2, 0, 0), SUNIrrep(2, 1, 0), SUNIrrep(3, 0, 0),
              SUNIrrep(3, 1, 0), SUNIrrep(4, 0, 0))
    dims = (3, 6, 8, 10, 15, 15)
    indices = (1, 5, 6, 15, 20, 35)
    names = ("3", "6", "8", "10", "15", "15′")
    for (i, I) in enumerate(irreps)
        @test dynkin_label(I) == (I.I[1] - I.I[2], I.I[2] - I.I[3])
        @test congruency(I) == (I.I[1] + I.I[2] + I.I[3]) % 3
        @test index(I) == indices[i]
    end
end
