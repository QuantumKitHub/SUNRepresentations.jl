println("------------------------------------")
println("Caching tests")
println("------------------------------------")

# Tests for caching of Clebsch-Gordan coefficients
import SUNRepresentations: cache_info, precompute_disk_cache, clear_disk_cache!

# only remove cache if running on CI
if get(ENV, "CI", false) == "true"
    println("Detected running on CI")
    clear_disk_cache!()
end

L = length(sprint(SUNRepresentations.cache_info))
for N in 3:4
    precompute_disk_cache(N, 1)
    L′ = length(sprint(cache_info))
    @test L′ >= L
    global L = L′
end

if get(ENV, "CI", false) == "true"
    println("Detected running on CI")
    clear_disk_cache!(3, Float64)
end

@testset "RAM cache info" begin
    # https://github.com/QuantumKitHub/SUNRepresentations.jl/issues/43
    # info should be shown even when the cache is empty
    empty!(SUNRepresentations.CGC_CACHE)
    str = sprint(SUNRepresentations.ram_cache_info)
    @test occursin("currentsize=0", str)
    @test occursin("maxsize=$(SUNRepresentations.CGC_CACHE.maxsize)", str)
end

@testset "Optional disk cache" begin
    old = SUNRepresentations.use_disk_cache()
    try
        @test SUNRepresentations.use_disk_cache(false) == old
        @test !SUNRepresentations.use_disk_cache()
        @test occursin("disabled", sprint(SUNRepresentations.disk_cache_info))
        @test_throws InvalidStateException precompute_disk_cache(3, 1)

        # CGCs are unaffected by the disk cache being disabled
        s1 = s2 = SUNIrrep{3}(2, 1, 0)
        empty!(SUNRepresentations.CGC_CACHE)
        without = [copy(CGC(Float64, s1, s2, s3)) for s3 in s1 ⊗ s2]
        SUNRepresentations.use_disk_cache(true)
        empty!(SUNRepresentations.CGC_CACHE)
        with = [CGC(Float64, s1, s2, s3) for s3 in s1 ⊗ s2]
        @test all(splat(≈), zip(without, with))
    finally
        SUNRepresentations.use_disk_cache(old)
    end
end

@testset "Custom disk cache directory" begin
    mktempdir() do dir
        old = SUNRepresentations.cgc_cache_dir(dir)
        try
            @test old === nothing  # was using the default scratchspace
            @test SUNRepresentations.cgc_cache_dir() == abspath(dir)
            @test_throws ArgumentError SUNRepresentations.cgc_cache_dir("")

            # cache is populated in the custom directory and read back from it
            @test isempty(readdir(dir))
            precompute_disk_cache(3, 1)
            @test !isempty(readdir(dir))
            @test isdir(joinpath(dir, "3", "Float64"))
            @test occursin("SU(3)", sprint(SUNRepresentations.disk_cache_info))

            s1 = s2 = SUNIrrep{3}(2, 1, 0)
            reference = [copy(CGC(Float64, s1, s2, s3)) for s3 in s1 ⊗ s2]
            empty!(SUNRepresentations.CGC_CACHE)
            @test all(splat(≈), zip(reference, (CGC(Float64, s1, s2, s3) for s3 in s1 ⊗ s2)))

            # unrelated data in the custom directory is left alone
            touch(joinpath(dir, "keepme"))
            clear_disk_cache!()
            @test readdir(dir) == ["keepme"]
        finally
            SUNRepresentations.cgc_cache_dir(old)
            empty!(SUNRepresentations.CGC_CACHE)
        end
        return @test SUNRepresentations.cgc_cache_dir() != abspath(dir)
    end
end

@testset "Disk cache environment variable" begin
    old = SUNRepresentations.use_disk_cache()
    try
        for flag in (false, true)
            withenv("SUNREPRESENTATIONS_USE_DISK_CACHE" => string(flag)) do
                SUNRepresentations._init_use_disk_cache!()
                return @test SUNRepresentations.use_disk_cache() == flag
            end
        end
        # invalid values are ignored, falling back to the preference
        withenv("SUNREPRESENTATIONS_USE_DISK_CACHE" => "yes") do
            @test_logs (:warn,) SUNRepresentations._init_use_disk_cache!()
            return @test SUNRepresentations.use_disk_cache() ==
                SUNRepresentations.Preferences.load_preference(
                SUNRepresentations, "use_disk_cache", true
            )
        end
    finally
        SUNRepresentations.use_disk_cache(old)
    end
end
