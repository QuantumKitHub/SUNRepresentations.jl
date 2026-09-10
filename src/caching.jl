"""
    CGC_CACHE = LRU{Any,SparseArray{Float64,4}}(; maxsize=100_000)

Global cache for storing Clebsch-Gordan Coefficients.
"""
const CGC_CACHE = LRU{Any, SparseArray{Float64, 4}}(; maxsize = 100_000)

# convert sector to string key
_key(s::SUNIrrep) = string(weight(s))

# Disk cache switch
# -----------------

const _USE_DISK_CACHE = Ref{Bool}(true)

"""
    use_disk_cache() -> Bool
    use_disk_cache(flag::Bool; persist=false) -> Bool

Query or set whether Clebsch-Gordan coefficients are cached on disk. Setting the flag
returns the previous value.

When the disk cache is disabled, CGCs are never read from or written to disk and the
scratchspace is never created, so no file locks are taken. The in-memory [`CGC_CACHE`](@ref)
is unaffected, meaning coefficients are still reused within a session but have to be
recomputed in the next one.

The setting is only changed for the current session, unless `persist=true`, in which case it
is also stored in the active project's `LocalPreferences.toml`. Note that writing
preferences is not safe to do concurrently; on a cluster, prefer the
`SUNREPRESENTATIONS_USE_DISK_CACHE` environment variable, which is read when the package is
loaded and takes precedence over the stored preference.

See also [`cgc_cache_dir`](@ref).
"""
use_disk_cache() = _USE_DISK_CACHE[]
function use_disk_cache(flag::Bool; persist::Bool = false)
    old = _USE_DISK_CACHE[]
    _USE_DISK_CACHE[] = flag
    persist && @set_preferences!("use_disk_cache" => flag)
    return old
end

function _init_use_disk_cache!()
    val = get(ENV, "SUNREPRESENTATIONS_USE_DISK_CACHE", nothing)
    if !isnothing(val)
        flag = tryparse(Bool, lowercase(strip(val)))
        if isnothing(flag)
            @warn "Ignoring SUNREPRESENTATIONS_USE_DISK_CACHE=$(repr(val)), expected \"true\" or \"false\"."
        else
            _USE_DISK_CACHE[] = flag
            return nothing
        end
    end
    _USE_DISK_CACHE[] = @load_preference("use_disk_cache", true)
    return nothing
end

# Disk cache
# ----------

# an empty string means the default package-wide scratchspace
const _CGC_CACHE_DIR = Ref{String}("")

"""
    cgc_cache_dir() -> String
    cgc_cache_dir(path::Union{AbstractString,Nothing}; persist=false) -> Union{String,Nothing}

Query or set the directory that holds the CGC disk cache. By default this is a package-wide
scratchspace, which is created upon first use. Passing a `path` makes the cache live there
instead, while passing `nothing` restores the default. Setting the directory returns the
previous setting, again as a `path` or as `nothing` for the default, so that it can be
restored by feeding it back in.

The directory is not created until something is actually written to it, and switching
directories neither moves nor removes any coefficients that were already cached elsewhere.

The setting is only changed for the current session, unless `persist=true`, in which case it
is also stored in the active project's `LocalPreferences.toml`. Note that writing
preferences is not safe to do concurrently.

See also [`use_disk_cache`](@ref).
"""
function cgc_cache_dir()
    dir = _CGC_CACHE_DIR[]
    return isempty(dir) ? @get_scratch!("CGC") : dir
end
function cgc_cache_dir(path::Union{AbstractString, Nothing}; persist::Bool = false)
    if path isa AbstractString && isempty(path)
        throw(ArgumentError("Empty CGC cache directory, use `nothing` to restore the default."))
    end
    old = _CGC_CACHE_DIR[]
    _CGC_CACHE_DIR[] = isnothing(path) ? "" : abspath(expanduser(path))
    if persist
        if isnothing(path)
            @delete_preferences!("cgc_cache_dir")
        else
            @set_preferences!("cgc_cache_dir" => _CGC_CACHE_DIR[])
        end
    end
    return isempty(old) ? nothing : old
end

function _init_cgc_cache_dir!()
    _CGC_CACHE_DIR[] = @load_preference("cgc_cache_dir", "")
    return nothing
end

function cgc_cachepath(s1::SUNIrrep{N}, s2::SUNIrrep{N}, T = Float64) where {N}
    return joinpath(cgc_cache_dir(), string(N), string(T), _key(s1), _key(s2))
end

function tryread(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where {T, N}
    use_disk_cache() || return nothing
    fn = cgc_cachepath(s1, s2, T)
    isfile(fn * ".jld2") || return nothing

    return mkpidlock(fn * ".pid"; stale_age = _PID_STALE_AGE) do
        try
            return jldopen(fn * ".jld2", "r"; parallel_read = true) do file
                @debug "loaded CGC from disk: $s1 ⊗ $s2 → $s3"
                !haskey(file, _key(s3)) && return nothing
                return file[_key(s3)]::SparseArray{T, 4}
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

function generate_all_CGCs(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}) where {T, N}
    @debug "Generating CGCs: $s1 ⊗ $s2"
    CGCs = Dict(_key(s3) => CGC(T, s1, s2, s3) for s3 in s1 ⊗ s2)
    return CGCs
end

function generate_CGC(
        ::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N},
        s3::SUNIrrep{N}
    ) where {T, N}
    @debug "Generating CGCs: $s1 ⊗ $s2"
    CGCs = _CGC(T, s1, s2, s3)
    use_disk_cache() || return CGCs

    fn = cgc_cachepath(s1, s2, T)
    isdir(dirname(fn)) || mkpath(dirname(fn))

    ks3 = _key(s3)
    mkpidlock(fn * ".pid"; stale_age = _PID_STALE_AGE) do
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
function precompute_disk_cache(N, a_max::Int = 1, T::Type{<:Number} = Float64; force = false)
    use_disk_cache() || throw(
        InvalidStateException(
            "The CGC disk cache is disabled, enable it with `SUNRepresentations.use_disk_cache(true)`.",
            :disk_cache_disabled
        )
    )
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
    fldrname = joinpath(cgc_cache_dir(), string(N), string(T))
    if isdir(fldrname)
        @info "Removing disk cache SU($N): $T"
        rm(fldrname; recursive = true)
    end
    return nothing
end
function clear_disk_cache!(N)
    fldrname = joinpath(cgc_cache_dir(), string(N))
    if isdir(fldrname)
        @info "Removing disk cache SU($N)"
        rm(fldrname; recursive = true)
    end
    return nothing
end
function clear_disk_cache!()
    if isempty(_CGC_CACHE_DIR[])
        Scratch.clear_scratchspaces!(SUNRepresentations)
    else
        # a custom directory may hold unrelated data, so only remove the SU(N) subtrees
        dir = _CGC_CACHE_DIR[]
        isdir(dir) || return nothing
        for entry in readdir(dir; join = true)
            isdir(entry) && !isnothing(tryparse(Int, basename(entry))) &&
                rm(entry; recursive = true)
        end
    end
    return nothing
end

"""
    ram_cache_info([io=stdout])

Print information about the in-memory CGC cache to `io`.
"""
function ram_cache_info(io::IO = stdout)
    println(io, "CGC RAM cache info:")
    println(io, LRUCache.cache_info(CGC_CACHE))
    return nothing
end

"""
    disk_cache_info([io=stdout]; clean=false)

Print information about the CGC disk cache to `io`. If `clean=true`, remove any corrupted files.
"""
function disk_cache_info(io::IO = stdout; clean = false)
    if !use_disk_cache()
        println(io, "CGC disk cache is disabled.")
        return nothing
    end
    cache_dir = cgc_cache_dir()
    if !isdir(cache_dir) || isempty(readdir(cache_dir))
        println(io, "CGC disk cache is empty.")
        return nothing
    end
    println(io, "CGC disk cache info:")
    println(io, "====================")

    for fldr_N in readdir(cache_dir; join = true)
        isdir(fldr_N) || continue
        N = basename(fldr_N)
        for fldr_T in readdir(fldr_N; join = true)
            isdir(fldr_T) || continue
            T = basename(fldr_T)
            n_bytes = 0
            n_entries = 0
            for (root, _, files) in walkdir(fldr_T)
                for f in files
                    # wrap in try/catch to avoid stopping the loop if a file is corrupted
                    try
                        n_entries += jldopen(
                            file -> length(keys(file)), joinpath(root, f), "r"
                        )
                        n_bytes += filesize(joinpath(root, f))
                    catch e
                        println(io, "Error in file $(joinpath(root, f)) : $e")
                        clean && rm(joinpath(root, f); force = true)
                    end
                end
            end
            println(
                io,
                "* SU($N) - $T - $(n_entries) entries - $(Base.format_bytes(n_bytes))"
            )
        end
    end
    return nothing
end

"""
    cache_info([io=stdout])

Print information about the CGC cache.
"""
function cache_info(io::IO = stdout)
    ram_cache_info(io)
    println(io)
    disk_cache_info(io)
    return nothing
end
