@testset "casimir" begin
    # --- SU(2): C₂(m) = m(m+2)/4, all odd C_k = 0 ---
    for m in 0:10
        @test casimir(2, SUNIrrep{2}([m])) == m * (m + 2) // 4
        @test casimir(3, SUNIrrep{2}([m])) == 0
    end

    # --- SU(3): C₂(p,q) = (p²+q²+pq+3p+3q)/3, C₃(p,q) = (p-q)(2p+q+3)(p+2q+3)/18 ---
    for p in 0:10, q in 0:10
        irrep = SUNIrrep{3}([p, q])
        @test casimir(2, irrep) == (p^2 + q^2 + p * q + 3p + 3q) // 3
        @test casimir(3, irrep) == (p - q) * (2p + q + 3) * (p + 2q + 3) // 18
    end

    # --- k > 3: structural properties only ---
    # Singlet always 0
    for k in 2:5, N in 2:5
        @test casimir(k, one(SUNIrrep{N})) == 0 // 1
    end

    # Conjugation: C_k(dual(λ)) = (-1)^k * C_k(λ)
    for N in 2:5
        for irrep in Iterators.take(values(SUNIrrep{N}), 20)
            for k in 2:5
                @test casimir(k, dual(irrep)) == (-1)^k * casimir(k, irrep)
            end
        end
    end
end
