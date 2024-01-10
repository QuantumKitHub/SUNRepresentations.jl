const CGCKEY{N} = NTuple{3,SUNIrrep{N}}
_string(key::CGCKEY) = "$(key[1].I) ⊗ $(key[2].I) → $(key[3].I)"

struct CGCCache{N,T}
    data::LRU{CGCKEY{N},SparseArray{T,4}} # RAM cached CGC tensors
    function CGCCache{N,T}(; maxsize=10^5) where {N,T}
        data = LRU{CGCKEY{N},SparseArray{T,4}}(; maxsize)
        return new{N,T}(data)
    end
end

function Base.show(io::IO, cache::CGCCache{N,T}) where {N,T}
    println(io, typeof(cache))
    println(io, "    ", LRUCache.cache_info(cache.data))
    fn = cache_path(N, T)
    if isfile(fn)
        println(io, "    ", filesize(fn), " bytes on disk")
        jldopen(fn, "r") do file
            return println(io, "    $(length(keys(file))) entries in disk cache")
        end
    else
        println(io, "    no disk cache")
    end
end

# List of CGC caches for each N and T
const CGC_CACHES = LRU{Any,CGCCache}(; maxsize=10)

const CGC_CACHE_PATH = @get_scratch!("CGC")
cache_path(N, T=Float64) = joinpath(CGC_CACHE_PATH, "$(N)_$(T)")

function Base.get!(cache::CGCCache{N,T}, (s1, s2, s3)::CGCKEY{N})::SparseArray{T,4} where {T,N}
    return get!(cache.data, (s1, s2, s3)) do
        # if the key is not in the cache, check if it is in a file
        cachedir = joinpath(cache_path(N, T), "$(weight(s1))")
        isdir(cachedir) || mkpath(cachedir)
        fn = "$(weight(s1)) x $(weight(s2))"

        # try reading data
        if isfile(joinpath(cachedir, fn * ".jld2"))
            try
                return jldopen(joinpath(cachedir, fn * ".jld2"), "r";
                            parallel_read=true) do file
                    @debug "loaded CGC from disk: $s1 ⊗ $s2 → $s3"
                    return file[string(weight(s3))]::SparseArray{T,4}
                end
            catch
            end
        end
        # if failed, create new data
        CGCs = Dict(string(weight(s3′)) => _CGC(T, s1, s2, s3′)
                    for s3′ in s1 ⊗ s2)

        # write CGCs to disk
        mkpidlock(joinpath(cachedir, fn * ".pid")) do
            return save(joinpath(cachedir, fn * ".jld2"), CGCs)
        end

        # return CGC
        return CGCs[string(weight(s3))]
    end
end

"""
    precompute_disk_cache(N, a_max, [T=Float64]; force=false)

Populate the CGC cache for ``SU(N)`` with eltype `T` with all CGCs with Dynkin labels up to
``a_max``.
Will not recompute CGCs that are already in the cache, unless ``force=true``.
"""
function precompute_disk_cache(N, a_max::Int=3, T::Type{<:Number}=Float64; force=false)
    all_dynkinlabels = CartesianIndices(ntuple(_ -> (a_max + 1), N - 1))
    all_irreps = [SUNIrrep(reverse(cumsum(I.I .- 1))..., 0) for I in all_dynkinlabels]

    @sync begin
        for s1 in all_irreps
            cachedir = joinpath(cache_path(N, T), "$(weight(s1))")
            isdir(cachedir) || mkpath(cachedir)
            for s2 in all_irreps
                if force || !isfile(joinpath(cachedir, "$(weight(s1)) x $(weight(s2)).jld2"))
                    Threads.@spawn _compute_disk_cache($s1, $s2, $T)
                end
            end
        end
    end
    
    cache_info()
    return nothing
end

function _compute_disk_cache(s1::SUNIrrep{N}, s2::SUNIrrep{N}, T) where {N}
    @info "Computing CGC: $s1 ⊗ $s2"
    cachedir = joinpath(cache_path(N, T), "$(weight(s1))")
    fn = "$(weight(s1)) x $(weight(s2))"
    CGCs = Dict(string(weight(s3)) => _CGC(T, s1, s2, s3)
                for s3 in s1 ⊗ s2)

    # write CGCs to disk
    mkpidlock(joinpath(cachedir, fn * ".pid")) do
        return save(joinpath(cachedir, fn * ".jld2"), CGCs)
    end
end

"""
    clear_disk_cache!(N, [T=Float64])

Remove the CGC cache for ``SU(N)`` with eltype `T` from disk.
"""
function clear_disk_cache!(N, T=Float64)
    fn = cache_path(N, T)
    if isfile(fn)
        @info "Removing disk cache SU($N): $T"
        rm(fn)
    end
    return nothing
end
function clear_disk_cache!()
    Scratch.clear_scratchspaces!(SUNRepresentations)
    return nothing
end

_parse_filename(fn) = split(splitext(basename(fn))[1], "_")

"""
    cache_info()

Print information about the CGC cache.
"""
function cache_info(io::IO=stdout)
    println(io, "CGC RAM cache info:")
    println(io, "===================")
    for ((N, T), cache) in CGC_CACHES
        println(io, "SU($N) - $T")
        println(io, "------------------------")
        println(io, cache)
        println(io)
    end

    println(io)
    println(io, "CGC disk cache info:")
    println(io, "====================")

    cache_dir = CGC_CACHE_PATH
    isdir(cache_dir) || return nothing

    for fldr in readdir(cache_dir; join=true)
        isdir(fldr) || continue
        N, T = _parse_filename(fldr)

        n_bytes = 0
        n_entries = 0

        for (root, _, files) in walkdir(fldr)
            for f in files
                n_bytes += filesize(joinpath(root, f))
                n_entries += jldopen(file -> length(keys(file)), joinpath(root, f), "r")
            end
        end

        println(io, "SU($N) - $T")
        println(io, "    * ", n_entries, " entries")
        println(io, "    * ", Base.format_bytes(n_bytes))
        println(io)
    end
    return nothing
end
