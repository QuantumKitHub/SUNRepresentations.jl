# This is type piracy:
const SU₃ = SU{3}
const SU₄ = SU{4}
const SU₅ = SU{5}
TensorKitSectors.type_repr(::Type{SU₃}) = "SU₃"
TensorKitSectors.type_repr(::Type{SU₄}) = "SU₄"
TensorKitSectors.type_repr(::Type{SU₅}) = "SU₅"
Base.getindex(::TensorKitSectors.IrrepTable, ::Type{SU{N}}) where {N} = SUNIrrep{N, N - 1}

Base.IteratorSize(::Type{SectorValues{T}}) where {T <: SUNIrrep} = Base.IsInfinite()

Base.iterate(iter::SectorValues{<:SUNIrrep}, i::Int = 1) = iter[i], i + 1

# linear order of sectors: use manhattan ordering of dynkin labels (0-based)
# TODO: all sizes are infinite so manhattan indexing can be sped up if need be
function Base.getindex(::SectorValues{<:SUNIrrep{N}}, i::Int) where {N}
    sz = ntuple(Returns(typemax(Int)), N - 1)
    I = TensorKitSectors.manhattan_to_multidimensional_index(i, sz)
    dk_label = I .- 1 # dynkin labels are 0-based
    return SUNIrrep{N, N - 1}(map(UInt8, dk_label))  # direct inner constructor, no validation needed
end
function TensorKitSectors.findindex(::SectorValues{I}, s::I) where {I <: SUNIrrep}
    a = dynkin_label(s) .+ 1
    sz = ntuple(Returns(typemax(Int)), rank(I) - 1)
    return TensorKitSectors.to_manhattan_index(a, sz)
end

TensorKitSectors.dual(s::SUNIrrep) = typeof(s)(reverse(getfield(s, :a)))
TensorKitSectors.unit(::Type{I}) where {I <: SUNIrrep} = I(ntuple(Returns(zero(UInt8)), Val(rank(I) - 1)))

TensorKitSectors.FusionStyle(::Type{<:SUNIrrep}) = GenericFusion()
TensorKitSectors.BraidingStyle(::Type{<:SUNIrrep}) = Bosonic()

# `⊗` and `Nsymbol` are called very often, e.g. at every vertex of every fusion tree that is
# enumerated, and from many tasks at once, so they read the direct product from a cache that
# is read without locking: lookups go to an immutable snapshot, and an insertion (rare, there
# are few distinct pairs) publishes a new snapshot under a lock.
mutable struct SnapshotCache{K, V}
    @atomic dict::Dict{K, V}
    const lock::ReentrantLock
    const maxsize::Int
end
SnapshotCache{K, V}(; maxsize::Int = 10^5) where {K, V} =
    SnapshotCache{K, V}(Dict{K, V}(), ReentrantLock(), maxsize)

function cached!(f, c::SnapshotCache{K, V}, key) where {K, V}
    d = @atomic :acquire c.dict
    v = get(d, key, nothing)
    v === nothing || return v::V
    val = convert(V, f())
    @lock c.lock begin
        d = @atomic :acquire c.dict
        v = get(d, key, nothing)
        v === nothing || return v::V
        d′ = length(d) < c.maxsize ? copy(d) : Dict{K, V}()
        d′[key] = val
        @atomic :release c.dict = d′
    end
    return val
end
function Base.empty!(c::SnapshotCache{K, V}) where {K, V}
    @lock c.lock (@atomic :release c.dict = Dict{K, V}())
    return c
end
Base.length(c::SnapshotCache) = length(@atomic :acquire c.dict)

# An LRU cache split into shards by key hash, each with its own lock, such that concurrent
# lookups from many tasks rarely wait on each other.
struct ShardedLRU{K, V}
    shards::Vector{LRU{K, V}}
end
function ShardedLRU{K, V}(; maxsize::Int, nshards::Int = 64) where {K, V}
    return ShardedLRU{K, V}([LRU{K, V}(; maxsize = cld(maxsize, nshards)) for _ in 1:nshards])
end
@inline _shard(c::ShardedLRU, key) = @inbounds c.shards[(hash(key) % length(c.shards)) + 1]
cached!(f, c::ShardedLRU, key) = get!(f, _shard(c, key), key)
Base.empty!(c::ShardedLRU) = (foreach(empty!, c.shards); c)
Base.length(c::ShardedLRU) = sum(length, c.shards)

# one cache per rank (`SUNIrrep{N}`)
const DIRECTPRODUCT_CACHE = SnapshotCache{Int, Any}(; maxsize = typemax(Int))

# the returned dictionary is shared with the cache and must not be mutated
function _directproduct(s1::I, s2::I) where {I <: SUNIrrep}
    K = Tuple{I, I}
    V = Dict{I, Int}
    cache = cached!(() -> SnapshotCache{K, V}(), DIRECTPRODUCT_CACHE, rank(I))::SnapshotCache{K, V}
    return cached!(cache, (s1, s2)) do
        return directproduct(s1, s2)
    end
end

function TensorKitSectors.:⊗(s1::I, s2::I) where {I <: SUNIrrep}
    return SectorSet{I}(keys(_directproduct(s1, s2)))
end
function TensorKitSectors.Nsymbol(s1::I, s2::I, s3::I) where {I <: SUNIrrep}
    return get(_directproduct(s1, s2), s3, 0)
end
function TensorKitSectors.dim(s::SUNIrrep{N}) where {N}
    I = weight(s)
    dim = 1 // 1
    for k2 in 2:N, k1 in 1:(k2 - 1)
        dim *= (k2 - k1 + I[k1] - I[k2]) // (k2 - k1)
    end
    @assert denominator(dim) == 1
    return numerator(dim)
end

TensorKitSectors.sectorscalartype(::Type{<:SUNIrrep}) = Float64
TensorKitSectors.fusiontensor(s1::I, s2::I, s3::I) where {I <: SUNIrrep} =
    CGC(s1, s2, s3)

# one cache per rank (`SUNIrrep{N}`)
const FCACHE = SnapshotCache{Int, Any}(; maxsize = typemax(Int))

TensorKitSectors.fusionscalartype(::Type{<:SUNIrrep}) = Float64

function TensorKitSectors.Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I <: SUNIrrep}
    key = (a, b, c, d, e, f)
    K = typeof(key)
    V = Array{fusionscalartype(I), 4}
    cache = cached!(() -> ShardedLRU{K, V}(; maxsize = 10^5), FCACHE, rank(I))::ShardedLRU{K, V}
    return cached!(cache, key) do
        return _Fsymbol(a, b, c, d, e, f)
    end
end
function _Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I <: SUNIrrep}
    N1 = Nsymbol(a, b, e)
    N2 = Nsymbol(e, c, d)
    N3 = Nsymbol(b, c, f)
    N4 = Nsymbol(a, f, d)

    (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0) &&
        return fill(zero(fusionscalartype(I)), N1, N2, N3, N4)

    # computing first diagonal element
    A = fusiontensor(a, b, e)
    B = fusiontensor(e, c, d)[:, :, 1, :]
    C = fusiontensor(b, c, f)
    D = fusiontensor(a, f, d)[:, :, 1, :]

    @tensor F[-1, -2, -3, -4] := conj(D[1, 5, -4]) * conj(C[2, 4, 5, -3]) *
        A[1, 2, 3, -1] * B[3, 4, -2]
    return Array(F)
end

const RCACHE = SnapshotCache{Int, Any}(; maxsize = typemax(Int))
TensorKitSectors.braidingscalartype(::Type{<:SUNIrrep}) = Float64
function TensorKitSectors.Rsymbol(a::I, b::I, c::I) where {I <: SUNIrrep}
    key = (a, b, c)
    K = typeof(key)
    V = Array{braidingscalartype(I), 2}
    cache = cached!(() -> ShardedLRU{K, V}(; maxsize = 10^5), RCACHE, rank(I))::ShardedLRU{K, V}
    return cached!(cache, key) do
        return _Rsymbol(a, b, c)
    end
end
function _Rsymbol(a::I, b::I, c::I) where {I <: SUNIrrep}
    N1 = Nsymbol(a, b, c)
    N2 = Nsymbol(b, a, c)

    (N1 == 0 || N2 == 0) && return fill(zero(braidingscalartype(I)), N1, N2)

    A = fusiontensor(a, b, c)[:, :, 1, :]
    B = fusiontensor(b, a, c)[:, :, 1, :]

    @tensor R[-1; -2] := conj(B[1, 2, -2]) * A[2, 1, -1]
    return Array(R)
end
