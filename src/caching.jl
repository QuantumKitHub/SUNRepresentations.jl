const CGCKEY{N} = NTuple{3,SUNIrrep{N}}
_string(key::CGCKEY) = "$(key[1].I) ⊗ $(key[2].I) → $(key[3].I)"

struct CGCCache{N,T}
    data::LRU{CGCKEY{N},SparseArray{T,4}} # RAM cached CGC tensors
    filelock::ReentrantLock # lock for writing to disk
    function CGCCache{N,T}(; maxsize=10^5) where {N,T}
        data = LRU{CGCKEY{N},SparseArray{T,4}}(; maxsize)
        return new{N,T}(data, ReentrantLock())
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
cache_path(N, T=Float64) = joinpath(CGC_CACHE_PATH, "$(N)_$(T).jld2")

function Base.get!(f::Function, cache::CGCCache{N,T},
                   key::CGCKEY{N})::SparseArray{T,4} where {T,N}
    return get!(cache.data, key) do
        # if the key is not in the cache, check if it is in the file
        key_str = _string(key)
        fn = cache_path(N, T)
        lock(cache.filelock)
        try
            file = jldopen(fn, "a+")
            if haskey(file, key_str)
                CGC = file[key_str]::SparseArray{T,4}
                close(file)
                @debug "loaded CGC from disk: $key_str"
                return CGC
            end
            close(file)
        finally
            unlock(cache.filelock)
        end

        # if the key is not in the file, compute it
        CGC = f()
        lock(cache.filelock) do
            jldopen(fn, "a+") do file
                file[key_str] = CGC
                @debug "wrote CGC to disk: $key_str"
                return nothing
            end
        end

        return CGC
    end
end

"""
    precompute_disk_cache(N, a_max, [T=Float64])

Populate the CGC cache for ``SU(N)`` with eltype `T` with all CGCs with Dynkin labels up to
``a_max``.
"""
function precompute_disk_cache(N, a_max::Int=3, T::Type{<:Number}=Float64)
    all_dynkinlabels = CartesianIndices(ntuple(_ -> (a_max + 1), N - 1))
    all_irreps = [SUNIrrep(reverse(cumsum(I.I .- 1))..., 0) for I in all_dynkinlabels]
    l = ReentrantLock()
    jldopen(cache_path(N, T), "a+") do file
        key_list = Set(keys(file))
        @sync for s1 in all_irreps, s2 in all_irreps
            for s3 in Iterators.filter(s -> maximum(dynkin_labels(s)) < a_max, s1 ⊗ s2)
                key_str = _string((s1, s2, s3))
                if key_str ∉ key_list
                    Threads.@spawn begin
                        cgc = _CGC(T, s1, s2, s3)
                        lock(l) do
                            file[key_str] = cgc
                            @info "Wrote CGC to disk: $key_str"
                        end
                    end
                end
            end
        end
        return nothing
    end
    cache_info()
    return nothing
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
    for file in readdir(CGC_CACHE_PATH)
        if endswith(file, ".jld2")
            N, T = _parse_filename(file)
            @info "Removing disk cache SU($N): $T"
            rm(cache_path(N, T))
        end
    end
    return nothing
end

_parse_filename(fn) = split(splitext(basename(fn))[1], "_")

"""
    cache_info()

Print information about the CGC cache.
"""
function cache_info()
    println("CGC RAM cache info:")
    println("===================")
    for ((N, T), cache) in CGC_CACHES
        println("SU($N) - $T")
        println("------------------------")
        println(cache)
        println()
    end
    println()
    println("CGC disk cache info:")
    println("====================")
    cache_dir = CGC_CACHE_PATH
    for file in readdir(cache_dir)
        if endswith(file, ".jld2")
            N, T = _parse_filename(file)

            fsz = filesize(joinpath(cache_dir, file))
            nentries = jldopen(joinpath(cache_dir, file), "r") do file
                return length(keys(file))
            end

            println("SU($N) - $T")
            println("    * ", nentries, " entries")
            println("    * ", Base.format_bytes(fsz))
            println()
        end
    end
    return nothing
end
