# The Clebsch-Gordan coefficients, F- and R-symbols and direct products are cached with
# MemoizationKit, see `SUN_CGC`. The CGCs are also cached on disk, in a scratch space of
# SUNRepresentations by default. Both are configured through MemoizationKit's preferences, e.g.
# `MemoizationKit.set_cache_preferences!(SUNRepresentations; disk_path = "/scratch/cgc")`.
# The RAM caches are bounded in bytes: 4 GiB for the CGCs, 512 MiB for the F-symbols and
# 128 MiB for the R-symbols.

"""
    precompute_disk_cache(N, a_max = 1, T = Float64)

Compute and cache, in RAM and on disk, all Clebsch-Gordan coefficients with eltype `T` for
`s1 ⊗ s2 → s3`, where `s1` and `s2` run over the irreps of ``SU(N)`` with Dynkin labels up to
`a_max`, and `s3` over the fusion products.
"""
function precompute_disk_cache(N, a_max::Int = 1, T::Type{<:Number} = Float64)
    irreps = collect(all_dynkin(SUNIrrep{N}, a_max))
    @sync for s1 in irreps, s2 in irreps
        Threads.@spawn foreach(s3 -> CGC(T, s1, s2, s3), s1 ⊗ s2)
    end
    return nothing
end

"""
    cache_info([io = stdout])

Print information about the caches of SUNRepresentations, in RAM and on disk.
"""
function cache_info(io::IO = stdout)
    for f in (SUN_CGC, SUN_Fsymbol, SUN_Rsymbol, productirreps)
        caches = MemoizationKit.cache_info(f)
        isempty(caches) && println(io, f, ": not used yet")
        for (_, c) in caches
            println(io, f, ": ", c)
        end
    end
    for (f, info) in MemoizationKit.disk_cache_info(SUN_CGC)
        println(io, f, " on disk: ", info.entries, " entries, ", Base.format_bytes(info.bytes), ", ", info.path)
    end
    return nothing
end
