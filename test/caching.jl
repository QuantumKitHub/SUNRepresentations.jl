# Tests for caching of Clebsch-Gordan coefficients
import SUNRepresentations: cache_info, precompute_disk_cache, cache_path, offsets_path, clear_disk_cache!

for N in 3:5
    cache_info()
    precompute_disk_cache(N, 1)
    @test isfile(cache_path(N))
    @test isfile(offsets_path(N))
    clear_disk_cache!(N)
    @test !isfile(cache_path(N))
    @test !isfile(offsets_path(N))
end

for N in 3:5
    precompute_disk_cache(N, 2)
end
cache_info()
