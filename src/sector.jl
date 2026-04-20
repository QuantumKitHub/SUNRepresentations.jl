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

function TensorKitSectors.:⊗(s1::I, s2::I) where {I <: SUNIrrep}
    return SectorSet{I}(keys(directproduct(s1, s2)))
end
function TensorKitSectors.Nsymbol(s1::I, s2::I, s3::I) where {I <: SUNIrrep}
    return get(directproduct(s1, s2), s3, 0)
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

const FCACHE = LRU{Int, Any}(; maxsize = 10)

TensorKitSectors.fusionscalartype(::Type{<:SUNIrrep}) = Float64

function TensorKitSectors.Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I <: SUNIrrep}
    key = (a, b, c, d, e, f)
    K = typeof(key)
    V = Array{fusionscalartype(I), 4}
    cache::LRU{K, V} = get!(FCACHE, rank(I)) do
        return LRU{K, V}(; maxsize = 10^5)
    end
    return get!(cache, key) do
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

const RCACHE = LRU{Int, Any}(; maxsize = 10)
TensorKitSectors.braidingscalartype(::Type{<:SUNIrrep}) = Float64
function TensorKitSectors.Rsymbol(a::I, b::I, c::I) where {I <: SUNIrrep}
    key = (a, b, c)
    K = typeof(key)
    V = Array{braidingscalartype(I), 2}
    cache::LRU{K, V} = get!(RCACHE, rank(I)) do
        return LRU{K, V}(; maxsize = 10^5)
    end
    return get!(cache, key) do
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
