const CGCKEY{N} = NTuple{3,SUNIrrep{N}}
const CGCCache{N,T} = LRU{CGCKEY{N},SparseArray{T,4}}

# convert sector to string key
_key(s::SUNIrrep) = string(weight(s))

# List of CGC caches for each N and T
const CGC_CACHES = LRU{Any,CGCCache}(; maxsize=10)

const CGC_CACHE_PATH = @get_scratch!("CGC")
function cgc_cachepath(s1::SUNIrrep{N}, s2::SUNIrrep{N}, T=Float64) where {N}
    return joinpath(CGC_CACHE_PATH, string(N), string(T), _key(s1), _key(s2))
end

function tryread(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where {T,N}
    fn = cgc_cachepath(s1, s2, T) * ".jld2"
    isfile(fn) || return nothing

    try
        return jldopen(fn, "r"; parallel_read=true) do file
            @debug "loaded CGC from disk: $s1 ⊗ $s2 → $s3"
            return file[_key(s3)]::SparseArray{T,4}
        end
    catch
    end

    return nothing
end

#= wait at most 1 min before deciding to overwrite. This should avoid deadlocking if a
process started writing but got killed before removing the pidfile. =#
const _PID_STALE_AGE = 60.0

function generate_all_CGCs(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}) where {T,N}
    @debug "Generating CGCs: $s1 ⊗ $s2"
    CGCs = Dict(_key(s3) => _CGC(T, s1, s2, s3) for s3 in s1 ⊗ s2)
    fn = cgc_cachepath(s1, s2, T)
    isdir(dirname(fn)) || mkpath(dirname(fn))

    mkpidlock(fn * ".pid"; stale_age=_PID_STALE_AGE) do
        return save(fn * ".jld2", CGCs)
    end

    return CGCs
end

"""
    precompute_disk_cache(N, a_max, [T=Float64]; force=false)

Populate the CGC cache for ``SU(N)`` with eltype `T` with all CGCs with Dynkin labels up to
``a_max``.
Will not recompute CGCs that are already in the cache, unless ``force=true``.
"""
function precompute_disk_cache(N, a_max::Int=1, T::Type{<:Number}=Float64; force=false)
    all_irreps = all_dynkin(SUNIrrep{N}, a_max)
    @sync for s1 in all_irreps, s2 in all_irreps
        if force || !isfile(cgc_cachepath(s1, s2, T) * ".jld2")
            Threads.@spawn begin
                generate_all_CGCs(T, s1, s2)
                nothing
            end
        end
    end

    cache_info()
    return nothing
end

"""
    clear_disk_cache!(N, [T=Float64])

Remove the CGC cache for ``SU(N)`` with eltype `T` from disk.
"""
function clear_disk_cache!(N, T=Float64)
    fldrname = joinpath(CGC_CACHE_PATH, string(N), string(T))
    if isdir(fldrname)
        @info "Removing disk cache SU($N): $T"
        rm(fldrname; recursive=true)
    end
    return nothing
end
function clear_disk_cache!()
    Scratch.clear_scratchspaces!(SUNRepresentations)
    return nothing
end

_parse_filename(fn) = split(splitext(basename(fn))[1], "_")

function ram_cache_info(io::IO=stdout)
    println(io, "CGC RAM cache info:")
    println(io, "===================")
    for ((N, T), cache) in CGC_CACHES
        println(io, "SU($N) - $T")
        println(io, "------------------------")
        println(io, "* ", LRUCache.cache_info(cache))
        println(io)
    end
    return nothing
end

function disk_cache_info(io::IO=stdout)
    println(io, "CGC disk cache info:")
    println(io, "====================")
    isdir(CGC_CACHE_PATH) || return nothing

    for fldr_N in readdir(CGC_CACHE_PATH; join=true)
        isdir(fldr_N) || continue
        N = last(splitpath(fldr_N))
        for fldr_T in readdir(fldr_N; join=true)
            isdir(fldr_T) || continue
            T = basename(fldr_T)
            n_bytes = 0
            n_entries = 0
            for (root, _, files) in walkdir(fldr_T)
                for f in files
                    n_bytes += filesize(joinpath(root, f))
                    n_entries += jldopen(file -> length(keys(file)), joinpath(root, f), "r")
                end
            end
            println(io,
                    "* SU($N) - $T - $(n_entries) entries - $(Base.format_bytes(n_bytes))")
        end
    end
    return nothing
end

"""
    cache_info([io=stdout])

Print information about the CGC cache.
"""
function cache_info(io::IO=stdout)
    ram_cache_info(io)
    disk_cache_info(io)
    return nothing
end
