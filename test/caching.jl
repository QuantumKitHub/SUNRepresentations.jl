# Tests for caching of Clebsch-Gordan coefficients
for N in 3:5
    SUNRepresentations.cache_info()
    SUNRepresentations.precompute_disk_cache(N, 1)
    @test isfile(SUNRepresentations.cache_path(N))
    @test isfile(SUNRepresentations.offsets_path(N))
    SUNRepresentations.clear_disk_cache!(N)
    @test !isfile(SUNRepresentations.cache_path(N))
    @test !isfile(SUNRepresentations.offsets_path(N))
end

for N in 3:5
    SUNRepresentations.precompute_disk_cache(N, 2)
end
SUNRepresentations.cache_info()