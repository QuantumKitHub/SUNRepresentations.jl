struct CGCCache{T,N}
    data::LRU{NTuple{3,SUNIrrep{N}},SparseArray{T,4}} # cached CGC tensors
    offsets::Dict{NTuple{3,SUNIrrep{N}},Tuple{UInt,UInt}} # existing hash to file positions

    function CGCCache{T,N}(; maxsize=10^5) where {T,N}
        data = LRU{NTuple{3,SUNIrrep{N}},SparseArray{T,4}}(; maxsize)
        offsets = if isfile(offsets_path(N, T))
            deserialize(offsets_path(N, T))
        else
            Dict{NTuple{3,SUNIrrep{N}},Tuple{UInt,UInt}}()
        end
        return new{T,N}(data, offsets)
    end
end

# List of CGC caches for each N and T
const CGC_CACHES = LRU{Any,CGCCache}(; maxsize=10)

const CGC_CACHE_PATH = @get_scratch!("CGC")
offsets_path(N, T=Float64) = joinpath(CGC_CACHE_PATH, "$N-$T-offsets.bin")
cache_path(N, T=Float64) = joinpath(CGC_CACHE_PATH, "$N-$T.bin")

function Base.get!(f::Function, cache::CGCCache{T, N}, key::NTuple{3,SUNIrrep{N}}) where {T, N}
    return get!(cache.data, key) do
        # if the key is not in the cache, check if it is in the file
        if haskey(cache.offsets, key)
            start, stop = cache.offsets[key]
            fn = cache_path(N, T)
            return open(fn, "r") do io
                return load_disk_cache_entry(io, start, stop)::SparseArray{T,4}
            end
        else
            return f()
        end
    end
end

function load_disk_cache_entry(io::IO, start::UInt, stop::UInt)
    seek(io, start)
    buf = read(io, stop - start)
    return deserialize(IOBuffer(buf))
end

function store_disk_cache_entry(io::IO, start::UInt, val)
    seek(io, start)
    buf = IOBuffer()
    serialize(buf, val)
    posbuf = UInt(position(buf))
    write(io, buf.data[1:posbuf])
    return UInt(position(io))
end

"""
    sync_disk_cache(N, [T=Float64])

Sync the CGC cache for ``SU(N)`` with eltype `T` to disk.

!!! warning
    This function is not thread-safe. In particular, it is not safe to call this function while other threads are accessing the cache.
"""
sync_disk_cache(N, T=Float64) = sync_disk_cache(CGC_CACHES[(N, T)])
function sync_disk_cache(cache::CGCCache{T, N}) where {T,N}
    # store data
    open(cache_path(N, T), "a+") do fid
        seekend(fid)
        for key in setdiff(keys(cache.data), keys(cache.offsets))
            start = UInt(position(fid))
            stop = store_disk_cache_entry(fid, start, cache.data[key])
            cache.offsets[key] = (start, stop)
        end
    end
    
    # store offsets
    open(offsets_path(N, T), "w") do fid
        serialize(fid, cache.offsets)
    end
end

"""
    clear_disk_cache(N, T)

Clear the CGC cache for ``SU(N)`` with eltype `T` from disk.
"""
clear_disk_cache!(N, T=Float64) = clear_disk_cache!(CGC_CACHES[(N, T)])
function clear_disk_cache!(cache::CGCCache{T, N}) where {T,N}
    isfile(cache_path(N, T)) && rm(cache_path(N, T))
    isfile(offsets_path(N, T)) && rm(offsets_path(N, T))
    empty!(cache.offsets)
    return nothing
end

"""
    clear_ram_cache!(N, T; sync=true)

Clear the CGC cache for ``SU(N)`` with eltype `T` from RAM. If `sync` is `true`, the cache will first be synced to disk.
"""
clear_ram_cache!(N, T=Float64; sync::Bool=true) = clear_ram_cache!(CGC_CACHES[(N, T)]; sync)
function clear_ram_cache!(cache::CGCCache; sync::Bool=true)
    sync && sync_disk_cache(cache)
    empty!(cache.data)
    return nothing
end

"""
    cache_info()

Print information about the CGC caches.
"""
function cache_info()
    # print RAM caches
    if isempty(CGC_CACHES)
        @info "CGC cache is empty"
    else
        for ((N, T), val) in CGC_CACHES
            sz = Base.summarysize(val) / 1024^2
            @info "CGC cache for SU($N) with eltype $T contains $(val.data.currentsize) / $(val.data.maxsize) CGCs ($sz Mib)"
        end
    end
    
    # print disk caches
    if !isdir(CGC_CACHE_PATH)
        @info "CGC cache directory $(CGC_CACHE_PATH) is empty"
    else
        for fn in readdir(CGC_CACHE_PATH)
            if endswith(fn, ".bin") && !endswith(fn, "offsets.bin")
                sz = filesize(joinpath(CGC_CACHE_PATH, fn)) / 1024^2
                num = length(deserialize(joinpath(CGC_CACHE_PATH, fn[1:end-4] * "-offsets.bin")))
                @info "CGC cache file $fn contains $num CGCs ($sz Mib)"
            end
        end
    end
    return nothing
end

"""
    precompute_disk_cache(N, a_max, [T=Float64])

Populate the CGC cache for ``SU(N)`` with eltype `T` with all CGCs with Dynkin labels up to
``a_max``.

See also: [`sync_disk_cache`](@ref)
"""
function precompute_disk_cache(N, a_max::Int=3, T::Type{<:Number}=Float64)
    all_dynkinlabels = CartesianIndices(ntuple(_ -> (a_max + 1), N - 1))
    Threads.@threads :dynamic for (I₁, I₂) in collect(Iterators.product(all_dynkinlabels, all_dynkinlabels))
        s1 = SUNIrrep(reverse(cumsum(I₁.I .- 1))..., 0)
        s2 = SUNIrrep(reverse(cumsum(I₂.I .- 1))..., 0)
        for s3 in s1 ⊗ s2
            if maximum(dynkin_labels(s3)) <= a_max
                Δt = @elapsed CGC(T, s1, s2, s3)
                @info "$(Threads.threadid()) computed $(s1.I) ⊗ $(s2.I) → $(s3.I) ($Δt sec)"
            end
        end
    end
    clear_ram_cache!(N, T; sync=true)
    return nothing
end
