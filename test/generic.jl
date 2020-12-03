println("------------------------------------")
println("Generic Irrep tests")
println("------------------------------------")
@timedtestset "Basic tests for Irrep{$N}:" for N = 2:5
    I1 = Irrep(tuple(sort(rand(1:9, N); rev=true)..., 1))
    I2 = Irrep(tuple(sort(rand(1:9, N); rev=true)..., 1))
    @constinferred dimension(I1)
    d = 0
    for (I, NI) in @constinferred directproduct(I1, I2)
        d += NI*dimension(I)
    end
    @test d == dimension(I1)*dimension(I2)
    if dimension(I2) < dimension(I1)
        I1, I2 = I2, I1
    end
    b = @constinferred basis(I1)
    (v, s) = @constinferred Nothing iterate(b)
    @constinferred Nothing iterate(b, s)
    v = @constinferred collect(b)
    for i = 1:length(v)-1
        @test isless(v[i], v[i+1])
    end
    @test v[end] == highest_weight(I1)
    s = sprint(show, v[1])
    for k = 1:N
        @test parse(Int, s[17 + 2*k]) == weight(I1)[k]
    end
end
