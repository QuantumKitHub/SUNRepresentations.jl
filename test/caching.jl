println("------------------------------------")
println("Caching tests")
println("------------------------------------")

# Tests for caching of Clebsch-Gordan coefficients
import SUNRepresentations: cache_info, precompute_disk_cache, cache_path, clear_disk_cache!

# only remove cache if running on CI
if get(ENV, "CI", false) == "true"
    println("Detected running on CI")
    clear_disk_cache!()
end

L = length(sprint(cache_info))
for N in 3:4
    precompute_disk_cache(N, 1)
    L′ = length(sprint(cache_info))
    @test L′ >= L
    global L = L′
end
