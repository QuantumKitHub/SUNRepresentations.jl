#= 
Wait at most 1 min before deciding to overwrite.
This should avoid deadlocking if a process started writing but got killed before removing the pidfile.
=#
"""
    const _PID_STALE_AGE = 60.0

Timeout for stale PID files in seconds.
"""
const _PID_STALE_AGE = 60.0

# convert sector to string key
_key(s::SUNIrrep) = string(weight(s))

# Clebsch-Gordan coefficients
# ---------------------------
"""
    CGC_CACHE = LRU{Any,SparseArray{Float64,4}}(; maxsize=100_000)

Global cache for storing Clebsch-Gordan Coefficients.
"""
const CGC_CACHE = LRU{Any,SparseArray{Float64,4}}(; maxsize=100_000)
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

function generate_all_CGCs(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}) where {T,N}
    @debug "Generating CGCs: $s1 ⊗ $s2"
    CGCs = Dict(_key(s3) => CGC(T, s1, s2, s3) for s3 in s1 ⊗ s2)
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

# F symbols
# ---------

"""
    F_CACHE = LRU{Any,Array{Float64,4}}(; maxsize=100_000)

Global cache for storing F-symbols.
"""
const F_CACHE = LRU{Any,Array{Float64,4}}(; maxsize=100_000)
const F_CACHE_PATH = @get_scratch!("Fsymbol")
_fkey(e, f) = "$(_key(e))_$(_key(f))"

# TODO: verify that this file doesn't become too large
function f_cachepath(a::I, b::I, c::I, d::I) where {N,I<:SUNIrrep{N}}
    return joinpath(F_CACHE_PATH, string(N), join(_key.((a, b, c, d)), '_'))
end

function tryread_F(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N},
                   d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where {N}
    fn = f_cachepath(a, b, c, d)
    isfile(fn * ".jld2") || return nothing

    F = mkpidlock(fn * ".pid"; stale_age=_PID_STALE_AGE) do
        try
            return jldopen(fn * ".jld2", "r"; parallel_read=true) do file
                !haskey(file, _fkey(e, f)) && return nothing
                return file[_fkey(e, f)]::Array{Float64,4}
            end
        catch
            return nothing
        end
    end
    isnothing(F) || @debug "loaded Fsymbol from disk: $a $b $c $d"
    return F
end

function generate_F(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N},
                    d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where {N}
    @debug "Generating Fsymbol: $a $b $c $d $e $f"
    F = _Fsymbol(a, b, c, d, e, f)
    fn = f_cachepath(a, b, c, d)
    isdir(dirname(fn)) || mkpath(dirname(fn))

    key = _fkey(e, f)
    mkpidlock(fn * ".pid"; stale_age=_PID_STALE_AGE) do
        return jldopen(fn * ".jld2", "a+") do file
            if !haskey(file, key)
                file[key] = F
            end
        end
    end

    return F
end

function generate_all_Fs(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N},
                         d::SUNIrrep{N}) where {N}
    @debug "Generating all Fs: $a $b $c $d"
    es = collect(intersect(⊗(a, b), map(dual, ⊗(c, dual(d)))))
    fs = collect(intersect(⊗(b, c), map(dual, ⊗(dual(d), a))))
    Fs = Dict(_fkey(e, f) => Fsymbol(a, b, c, d, e, f) for e in es, f in fs)
    return Fs
end

"""
    precompute_disk_cache_F(N, a_max, [T=Float64]; force=false)

Populate the disk cache for ``SU(N)`` with eltype `T` with all Fsymbols with Dynkin labels up to
``a_max``.
Will not recompute Fsymbols that are already in the cache, unless `force=true`.
"""
function precompute_disk_cache_F(N, a_max::Int=1; force=false)
    all_irreps = all_dynkin(SUNIrrep{N}, a_max)
    for a in all_irreps, b in all_irreps, c in all_irreps
        for d in ⊗(a, b, c)
            maximum(dynkin_label(d)) ≤ a_max || continue
            if force || !isfile(f_cachepath(a, b, c, d) * ".jld2")
                generate_all_Fs(a, b, c, d)
            end
        end
    end
    disk_cache_F_info()
    return nothing
end

# R symbols
# ---------

"""
    R_CACHE = LRU{Any,Matrix{Float64}}(; maxsize=100_000)

Global cache for storing R-symbols.
"""
const R_CACHE = LRU{Any,Matrix{Float64}}(; maxsize=100_000)
const R_CACHE_PATH = @get_scratch!("Rsymbol")

function r_cachepath(a::I, b::I) where {N,I<:SUNIrrep{N}}
    return joinpath(R_CACHE_PATH, string(N), join(_key.((a, b)), '_'))
end

function tryread_R(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where {N}
    fn = r_cachepath(a, b)
    isfile(fn * ".jld2") || return nothing

    R = mkpidlock(fn * ".pid"; stale_age=_PID_STALE_AGE) do
        try
            return jldopen(fn * ".jld2", "r"; parallel_read=true) do file
                !haskey(file, _key(c)) && return nothing
                return file[_key(c)]::Matrix{Float64}
            end
        catch
            return nothing
        end
    end
    isnothing(R) || @debug "loaded Rsymbol from disk: $a $b $c"
    return R
end

function generate_R(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where {N}
    @debug "Generating Rsymbol: $a $b $c"
    R = _Rsymbol(a, b, c)
    fn = r_cachepath(a, b)
    isdir(dirname(fn)) || mkpath(dirname(fn))

    key = _key(c)
    mkpidlock(fn * ".pid"; stale_age=_PID_STALE_AGE) do
        return jldopen(fn * ".jld2", "a+") do file
            if !haskey(file, key)
                file[key] = R
            end
        end
    end

    return R
end

function generate_all_Rs(a::SUNIrrep{N}, b::SUNIrrep{N}) where {N}
    @debug "Generating all Rs: $a $b"
    cs = ⊗(a, b)
    Rs = Dict(_key(c) => Rsymbol(a, b, c) for c in cs)
    return Rs
end

"""
    precompute_disk_cache_R(N, a_max, [T=Float64]; force=false)

Populate the disk cache for ``SU(N)`` with eltype `T` with all Rsymbols with Dynkin labels up to
``a_max``.
Will not recompute Rsymbols that are already in the cache, unless `force=true`.
"""
function precompute_disk_cache_R(N, a_max::Int=1; force=false)
    all_irreps = all_dynkin(SUNIrrep{N}, a_max)
    for a in all_irreps, b in all_irreps
        if force || !isfile(r_cachepath(a, b) * ".jld2")
            for c in ⊗(a, b)
                maximum(dynkin_label(c)) ≤ a_max || continue
                generate_R(a, b, c)
            end
        end
    end
    disk_cache_R_info()
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
    for (name, cache) in zip(("CGC", "F", "R"), (CGC_CACHE, F_CACHE, R_CACHE))
        if isempty(cache)
            println(io, "$name RAM cache is empty.")
        else
            info = LRUCache.cache_info(cache)
            println(io, "$name RAM cache info:")
            println(io, info)
        end
    end
    return nothing
end

"""
    disk_cache_info([io=stdout]; clean=false)

Print information about the CGC disk cache to `io`. If `clean=true`, remove any corrupted files.
"""
function disk_cache_info(io::IO=stdout; clean=false)
    disk_cache_CGC_info(io; clean)
    disk_cache_F_info(io; clean)
    disk_cache_R_info(io; clean)
    return nothing
end

function disk_cache_CGC_info(io::IO=stdout; clean=false)
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
            n_entries, n_bytes = summarize_folder(fldr_T; clean)
            println(io,
                    "* SU($N) - $T - $(n_entries) entries - $(Base.format_bytes(n_bytes))")
        end
    end
    println(io)
    return nothing
end

function disk_cache_F_info(io::IO=stdout; clean=false)
    if !isdir(F_CACHE_PATH) || isempty(readdir(F_CACHE_PATH))
        println("F disk cache is empty.")
        return nothing
    end
    println(io, "F disk cache info:")
    println(io, "==================")

    for fldr_N in readdir(F_CACHE_PATH; join=true)
        isdir(fldr_N) || continue
        N = basename(fldr_N)
        n_entries, n_bytes, n_files = summarize_folder(fldr_N; clean)
        println(io,
                "* SU($N) - $(n_files) files - $(n_entries) entries - $(Base.format_bytes(n_bytes))")
    end
    println(io)
    return nothing
end

function disk_cache_R_info(io::IO=stdout; clean=false)
    if !isdir(R_CACHE_PATH) || isempty(readdir(R_CACHE_PATH))
        println("R disk cache is empty.")
        return nothing
    end
    println(io, "R disk cache info:")
    println(io, "==================")

    for fldr_N in readdir(R_CACHE_PATH; join=true)
        isdir(fldr_N) || continue
        N = basename(fldr_N)
        n_entries, n_bytes, n_files = summarize_folder(fldr_N; clean)
        println(io,
                "* SU($N) - $(n_files) files - $(n_entries) entries - $(Base.format_bytes(n_bytes))")
    end
    println(io)
    return nothing
end

function summarize_folder(folder; clean=false)
    n_bytes = 0
    n_entries = 0
    n_files = 0
    for (root, _, files) in walkdir(folder)
        for f in files
            # wrap in try/catch to avoid stopping the loop if a file is corrupted
            try
                n_entries += jldopen(file -> length(keys(file)), joinpath(root, f), "r")
                n_bytes += filesize(joinpath(root, f))
                n_files += 1
            catch e
                println(io, "Error in file $(joinpath(root, f)) : $e")
                if clean
                    rm(joinpath(root, f); force=true)
                end
            end
        end
    end
    return n_entries, n_bytes, n_files
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
