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
end
