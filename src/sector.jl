# is this type piracy?
const SU₃ = SU{3}
const SU₄ = SU{4}
const SU₅ = SU{5}
TensorKitSectors.type_repr(::Type{SU₃}) = "SU₃"
TensorKitSectors.type_repr(::Type{SU₄}) = "SU₄"
TensorKitSectors.type_repr(::Type{SU₅}) = "SU₅"
Base.getindex(::TensorKitSectors.IrrepTable, ::Type{SU{N}}) where {N} = SUNIrrep{N}

Base.convert(::Type{SUNIrrep{N}}, I::NTuple{N,Int}) where {N} = SUNIrrep{N}(I)
Base.convert(::Type{SUNIrrep{N}}, I::Vector{Int}) where {N} = SUNIrrep{N}(I)
Base.convert(::Type{SUNIrrep{N}}, I::AbstractString) where {N} = SUNIrrep{N}(I)
function Base.IteratorSize(::Type{TensorKitSectors.SectorValues{T}}) where {T<:SUNIrrep}
    return Base.IsInfinite()
end

function Base.iterate(::TensorKitSectors.SectorValues{SUNIrrep{N}},
                      I=ntuple(zero, N)) where {N}
    s = SUNIrrep(I)
    k = N - 1
    while k > 1 && I[k] == I[k - 1]
        k -= 1
    end
    I = Base.setindex(I, I[k] + 1, k)
    for l in (k + 1):(N - 1)
        I = Base.setindex(I, 0, l)
    end
    return s, I
end

Base.:(==)(s::SUNIrrep, t::SUNIrrep) = ==(s.I, t.I)
Base.hash(s::SUNIrrep, h::UInt) = hash(s.I, h)
Base.conj(s::SUNIrrep) = SUNIrrep(s.I[1] .- reverse(s.I))
Base.one(::Type{SUNIrrep{N}}) where {N} = SUNIrrep(ntuple(n -> 0, N))

function TensorKitSectors.FusionStyle(::Type{SUNIrrep{N}}) where {N}
    return TensorKitSectors.GenericFusion()
end
Base.isreal(::Type{<:SUNIrrep}) = true
TensorKitSectors.BraidingStyle(::Type{<:SUNIrrep}) = TensorKitSectors.Bosonic()

function TensorKitSectors.:⊗(s1::SUNIrrep{N}, s2::SUNIrrep{N}) where {N}
    return TensorKitSectors.SectorSet{SUNIrrep{N}}(keys(directproduct(s1, s2)))
end
function TensorKitSectors.Nsymbol(s1::SUNIrrep{N}, s2::SUNIrrep{N},
                                  s3::SUNIrrep{N}) where {N}
    return get(directproduct(s1, s2), s3, 0)
end

function TensorKitSectors.fusiontensor(s1::SUNIrrep{N}, s2::SUNIrrep{N},
                                       s3::SUNIrrep{N}) where {N}
    return CGC(Float64, s1, s2, s3)
end

const FCACHE = LRU{Int,Any}(; maxsize=10)
function TensorKitSectors.Fsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N},
                                  d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where {N}
    key = (a, b, c, d, e, f)
    K = typeof(key)
    V = Array{Float64,4}
    cache::LRU{K,V} = get!(FCACHE, N) do
        return LRU{K,V}(; maxsize=10^5)
    end
    return get!(cache, key) do
        return _Fsymbol(a, b, c, d, e, f)
    end
end
function _Fsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N},
                  d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where {N}
    N1 = Nsymbol(a, b, e)
    N2 = Nsymbol(e, c, d)
    N3 = Nsymbol(b, c, f)
    N4 = Nsymbol(a, f, d)

    (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0) && return fill(0.0, N1, N2, N3, N4)

    # computing first diagonal element
    A = fusiontensor(a, b, e)
    B = fusiontensor(e, c, d)[:, :, 1, :]
    C = fusiontensor(b, c, f)
    D = fusiontensor(a, f, d)[:, :, 1, :]

    @tensor F[-1, -2, -3, -4] := conj(D[1, 5, -4]) * conj(C[2, 4, 5, -3]) *
                                 A[1, 2, 3, -1] * B[3, 4, -2]
    return Array(F)
end

const RCACHE = LRU{Int,Any}(; maxsize=10)
function TensorKitSectors.Rsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where {N}
    key = (a, b, c)
    K = typeof(key)
    V = Array{Float64,2}
    cache::LRU{K,V} = get!(RCACHE, N) do
        return LRU{K,V}(; maxsize=10^5)
    end
    return get!(cache, key) do
        return _Rsymbol(a, b, c)
    end
end
function _Rsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where {N}
    N1 = Nsymbol(a, b, c)
    N2 = Nsymbol(b, a, c)

    (N1 == 0 || N2 == 0) && return fill(0.0, N1, N2)

    A = fusiontensor(a, b, c)[:, :, 1, :]
    B = fusiontensor(b, a, c)[:, :, 1, :]

    @tensor R[-1; -2] := conj(B[1, 2, -2]) * A[2, 1, -1]
    return Array(R)
end
