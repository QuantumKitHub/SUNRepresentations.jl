println("------------------------------------")
println("Caching tests")
println("------------------------------------")

# Tests for caching of Clebsch-Gordan coefficients
import SUNRepresentations: cache_info, precompute_disk_cache, SUN_CGC, CGCSerializer
using Serialization: serialize, deserialize
using MemoizationKit: MemoizationKit, empty_caches!, empty_disk_caches!, disk_cache_info, uncached

disk_hits() = only(p.second.hits for p in MemoizationKit.disk_cache_stats() if p.first === SUN_CGC)

@testset "CGC disk cache" begin
    # only remove cache if running on CI
    get(ENV, "CI", "false") == "true" && empty_disk_caches!(SUN_CGC)

    precompute_disk_cache(3, 1)
    @test only(disk_cache_info(SUN_CGC)).second.entries > 0
    @test occursin("SUN_CGC", sprint(cache_info))

    # CGCs are read back from disk once they are no longer in RAM
    s1 = s2 = SUNIrrep{3}(2, 1, 0)
    reference = [uncached(SUN_CGC, Float64, s1, s2, s3) for s3 in s1 ⊗ s2]
    foreach(s3 -> CGC(Float64, s1, s2, s3), s1 ⊗ s2)
    empty_caches!(SUN_CGC)
    hits = disk_hits()
    @test all(splat(≈), zip(reference, (CGC(Float64, s1, s2, s3) for s3 in s1 ⊗ s2)))
    @test disk_hits() == hits + length(reference)
end

@testset "CGC serializer" begin
    s1 = s2 = SUNIrrep{3}(2, 1, 0)
    for s3 in s1 ⊗ s2, T in (Float64, Float32)
        C = CGC(Float64, s1, s2, s3)
        A = SparseArray{T, 4}(Dict(k => T(v) for (k, v) in C.data), size(C))
        io = IOBuffer()
        serialize(CGCSerializer(io), A)
        B = deserialize(CGCSerializer(seekstart(io)))
        @test B isa typeof(A) && size(B) == size(A) && B == A
    end
end
