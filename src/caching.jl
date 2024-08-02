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
    fn = cgc_cachepath(s1, s2, T)
    isfile(fn * ".jld2") || return nothing

    return mkpidlock(fn * ".pid"; stale_age=_PID_STALE_AGE) do
        try
            return jldopen(fn * ".jld2", "r"; parallel_read=true) do file
                @debug "loaded CGC from disk: $s1 ⊗ $s2 → $s3"
                !haskey(file, _key(s3)) && return nothing
                return file[_key(s3)]::SparseArray{T,4}
            end
        catch
        end
    end

    return nothing
end

#= 
Wait at most 1 min before deciding to overwrite.
This should avoid deadlocking if a process started writing but got killed before removing the pidfile.
=#
"""
    const _PID_STALE_AGE = 60.0

Timeout for stale PID files in seconds.
"""
const _PID_STALE_AGE = 60.0

function generate_all_CGCs(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}) where {T,N}
    @debug "Generating CGCs: $s1 ⊗ $s2"
    CGCs = Dict(begin
                    disked = tryread(T, s1, s2, s3)
                    if isnothing(disked)
                        _key(s3) => _CGC(T, s1, s2, s3)
                    else
                        _key(s3) => disked
                    end
                end
                for s3 in s1 ⊗ s2)
    fn = cgc_cachepath(s1, s2, T)
    isdir(dirname(fn)) || mkpath(dirname(fn))

    mkpidlock(fn * ".pid"; stale_age=_PID_STALE_AGE) do
        jldopen(fn * ".jld2", "a+") do file
            for (ks3, cgc) in CGCs
                if !haskey(file, ks3)
                    file[ks3] = cgc
                end
            end
        end
    end

    return CGCs
end

function generate_CGC(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N},
                      s3::SUNIrrep{N}) where {T,N}
    @debug "Generating CGCs: $s1 ⊗ $s2"
    CGCs = _CGC(T, s1, s2, s3)
    fn = cgc_cachepath(s1, s2, T)
    isdir(dirname(fn)) || mkpath(dirname(fn))

    ks3 = _key(s3)
    mkpidlock(fn * ".pid"; stale_age=_PID_STALE_AGE) do
        return jldopen(fn * ".jld2", "a+") do file
            if !haskey(file, ks3)
                file[ks3] = CGCs
            end
        end
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

    disk_cache_info()
    return nothing
end

"""
    clear_disk_cache!([N, [T]])

Remove the CGC cache for ``SU(N)`` with eltype `T` from disk. If the arguments are not
specified, this removes the cached CGCs for all values of that parameter.
"""
function clear_disk_cache!(N, T)
    fldrname = joinpath(CGC_CACHE_PATH, string(N), string(T))
    if isdir(fldrname)
        @info "Removing disk cache SU($N): $T"
        rm(fldrname; recursive=true)
    end
    return nothing
end
function clear_disk_cache!(N)
    fldrname = joinpath(CGC_CACHE_PATH, string(N))
    if isdir(fldrname)
        @info "Removing disk cache SU($N)"
        rm(fldrname; recursive=true)
    end
    return nothing
end
function clear_disk_cache!()
    Scratch.clear_scratchspaces!(SUNRepresentations)
    return nothing
end

function ram_cache_info(io::IO=stdout)
    if isempty(CGC_CACHES)
        println("CGC RAM cache is empty.")
        return nothing
    end
    println(io, "CGC RAM cache info:")
    println(io, "===================")
    for ((N, T), cache) in CGC_CACHES
        println(io, "* SU($N) - $T - $(LRUCache.cache_info(cache))")
    end
    return nothing
end

"""
    disk_cache_info([io=stdout]; clean=false)

Print information about the CGC disk cache to `io`. If `clean=true`, remove any corrupted files.
"""
function disk_cache_info(io::IO=stdout; clean=false)
    if !isdir(CGC_CACHE_PATH) || isempty(readdir(CGC_CACHE_PATH))
        println("CGC disk cache is empty.")
        return nothing
    end
    println(io, "CGC disk cache info:")
    println(io, "====================")

    for fldr_N in readdir(CGC_CACHE_PATH; join=true)
        isdir(fldr_N) || continue
        N = basename(fldr_N)
        for fldr_T in readdir(fldr_N; join=true)
            isdir(fldr_T) || continue
            T = basename(fldr_T)
            n_bytes = 0
            n_entries = 0
            for (root, _, files) in walkdir(fldr_T)
                for f in files
                    # wrap in try/catch to avoid stopping the loop if a file is corrupted
                    try
                        n_entries += jldopen(file -> length(keys(file)), joinpath(root, f),
                                             "r")
                        n_bytes += filesize(joinpath(root, f))
                    catch e
                        println(io, "Error in file $(joinpath(root, f)) : $e")
                        if clean
                            rm(joinpath(root, f); force=true)
                        end
                    end
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
    println(io)
    disk_cache_info(io)
    return nothing
end
