println("------------------------------------")
println("Caching tests")
println("------------------------------------")

# Tests for caching of Clebsch-Gordan coefficients
import SUNRepresentations: cache_info, precompute_disk_cache, cache_path, clear_disk_cache!
clear_disk_cache!()
for N in 3:4
    precompute_disk_cache(N, 1)
    @test isfile(cache_path(N))
end
