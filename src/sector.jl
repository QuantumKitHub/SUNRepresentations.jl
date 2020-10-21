using .TensorKit: dim, fusiontensor, Nsymbol, @tensor

export SUNIrrep

struct SUNIrrep{N} <: TensorKit.Irrep{TensorKit.SU{N}}
    I::NTuple{N,Int64}
end
SUNIrrep{N}(i::Vararg{Int64,N}) where N = SUNIrrep(i)

function Base.show(io::IO, c::SUNIrrep{N}) where {N}
    if get(io, :typeinfo, nothing) === typeof(c)
        print(io, c.I)
    else
        print(io, "SUNIrrep{$N}", c.I)
    end
end

Base.convert(::Type{SUNIrrep{N}}, i::Irrep) where N = SUNIrrep{N}(i.I)
Base.convert(::Type{SUNIrrep{N}}, I::NTuple{N,Int}) where N = SUNIrrep{N}(I)

Base.isless(s1::SUNIrrep{N}, s2::SUNIrrep{N}) where N = isless(Irrep(s1.I), Irrep(s2.I))

Base.IteratorSize(::Type{TensorKit.SectorValues{T}}) where T<:SUNIrrep = Base.IsInfinite()

function Base.iterate(::TensorKit.SectorValues{SUNIrrep{N}}, I = ntuple(zero, N)) where N
    s = SUNIrrep(I)
    k = N-1
    while k > 1 && I[k] == I[k-1]
        k -= 1
    end
    I = Base.setindex(I, I[k]+1, k)
    for l = k+1:N-1
        I = Base.setindex(I, 0, l)
    end
    return s, I
end

Base.:(==)(s::SUNIrrep, t::SUNIrrep) = ==(s.I, t.I)
Base.hash(s::SUNIrrep, h::UInt) = hash(s.I, h)

TensorKit.dim(s::SUNIrrep) = dimension(Irrep(s.I))
Base.conj(s::SUNIrrep) = SUNIrrep(s.I[1] .- reverse(s.I))
Base.one(::Type{SUNIrrep{N}}) where N = SUNIrrep(ntuple(n->0, N))

TensorKit.FusionStyle(::Type{SUNIrrep{N}}) where N = TensorKit.DegenerateNonAbelian()
Base.isreal(::Type{<:SUNIrrep}) = true
TensorKit.BraidingStyle(::Type{<:SUNIrrep}) = TensorKit.Bosonic();

TensorKit.:⊗(s1::SUNIrrep{N}, s2::SUNIrrep{N}) where N =
    TensorKit.SectorSet{SUNIrrep{N}}( keys(directproduct(Irrep(s1.I), Irrep(s2.I))) )
TensorKit.Nsymbol(s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where N =
    get(directproduct(Irrep(s1.I), Irrep(s2.I)), Irrep(s3.I), 0)

TensorKit.fusiontensor(s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where N =
    CGC(Float64, Irrep(s1.I), Irrep(s2.I), Irrep(s3.I))

const FCACHE = Vector{Any}()
function TensorKit.Fsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N},
                            d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where N
    key = (a, b, c, d, e, f)
    K = typeof(key)
    V = Array{Float64,4}
    if length(FCACHE) < N || !isassigned(FCACHE, N)
        resize!(FCACHE, max(length(FCACHE), N))
        cache = Dict{K,V}()
        FCACHE[N] = cache
    else
        cache::Dict{K,V} = FCACHE[N]
    end
    return get!(cache, key) do
        _Fsymbol(a,b,c,d,e,f)
    end
end
function _Fsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N},
                            d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where N
    N1 = Nsymbol(a,b,e)
    N2 = Nsymbol(e,c,d)
    N3 = Nsymbol(b,c,f)
    N4 = Nsymbol(a,f,d)

    (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0) && return fill(0.0, N1, N2, N3, N4)

    # computing first diagonal element
    A = fusiontensor(a,b,e)
    B = fusiontensor(e,c,d)[:, :, 1, :]
    C = fusiontensor(b,c,f)
    D = fusiontensor(a,f,d)[:, :, 1, :]

    @tensor F[-1,-2,-3,-4] := conj(D[1,5,-4]) * conj(C[2,4,5,-3]) *
                                A[1,2,3,-1] * B[3,4,-2]
    return Array(F)
end

const RCACHE = Vector{Any}()
function TensorKit.Rsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where N
    key = (a, b, c)
    K = typeof(key)
    V = Array{Float64,2}
    if length(RCACHE) < N || !isassigned(RCACHE, N)
        resize!(RCACHE, max(length(RCACHE), N))
        cache = Dict{K,V}()
        RCACHE[N] = cache
    else
        cache::Dict{K,V} = RCACHE[N]
    end
    return get!(cache, key) do
        _Rsymbol(a,b,c)
    end
end
function _Rsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where N
    N1 = Nsymbol(a, b, c);
    N2 = Nsymbol(b, a, c);

    (N1 == 0 || N2 == 0) && return fill(0.0, N1, N2)

    A = fusiontensor(a, b, c)[:, :, 1, :]
    B = fusiontensor(b, a, c)[:, :, 1, :]

    @tensor R[-1;-2] := conj(B[1, 2, -2]) * A[2, 1, -1]
    Array(R)
end
