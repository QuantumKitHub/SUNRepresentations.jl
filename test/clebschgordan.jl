println("------------------------------------")
println("Clebsch-Gordan coefficients")
println("------------------------------------")

using SUNRepresentations: annihilation
using SparseArrayKit: nonzero_values

# for SU(2), the CGCs are known exactly: compare with those of SU2Irrep, which has the opposite
# basis order, up to an overall sign (the largest spins have highest-weight CGCs below `TOL_GAUGE`)
@testset "SU(2) CGCs for large spins" begin
    for (n1, n2, n3) in ((40, 40, 20), (80, 80, 80), (80, 41, 61), (200, 200, 200), (250, 120, 200))
        s1, s2, s3 = SUNIrrep{2}(n1, 0), SUNIrrep{2}(n2, 0), SUNIrrep{2}(n3, 0)
        C = Array(CGC(Float64, s1, s2, s3))
        ref = fusiontensor(SU2Irrep(n1 // 2), SU2Irrep(n2 // 2), SU2Irrep(n3 // 2))
        ref = reverse(ref; dims = (1, 2, 3))
        @test isapprox(C, sign(LinearAlgebra.dot(C, ref)) * ref; atol = 1.0e-12)
    end
end

# CGCs are intertwiners: J⁻ C = C J⁻ for the lowering operators of all three irreps (the second
# case has outer multiplicity 4)
@testset "SU(3) CGCs are intertwiners" for irreps in (
        ((14, 0, 0), (14, 14, 0), (14, 7, 0)),
        ((6, 3, 0), (6, 3, 0), (6, 3, 0)),
    )
    s1, s2, s3 = map(SUNIrrep{3}, irreps)
    C = CGC(Float64, s1, s2, s3)
    for (J1, J2, J3) in zip(annihilation(s1), annihilation(s2), annihilation(s3))
        J1, J2, J3 = map(J -> SparseArray{Float64}(J), (J1, J2, J3))
        @tensor R[a, b, c, α] := J1[a, a′] * C[a′, b, c, α] + J2[b, b′] * C[a, b′, c, α] -
            C[a, b, c′, α] * J3[c′, c]
        @test maximum(abs, nonzero_values(R); init = 0.0) < 1.0e-12
    end
end
