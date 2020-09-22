import TensorKit
using TensorKit: dim, fusiontensor, Nsymbol, @tensor

export SUNIrrep

struct SUNIrrep{N} <: TensorKit.Irrep{TensorKit.SU{N}}
    I::NTuple{N,Int64}
end
SUNIrrep{N}(i::Vararg{Int64,N}) where N = SUNIrrep(i)

Base.isless(s1::SUNIrrep{N}, s2::SUNIrrep{N}) where N = isless(Irrep(s1.I), Irrep(s2.I))

Base.IteratorSize(::Type{TensorKit.SectorValues{T}}) where T<:SUNIrrep = Base.IsInfinite()

function Base.iterate(::TensorKit.SectorValues{T}, i = ntuple(x->0,Val{N}()))  where T<:SUNIrrep{N} where N
    ii = 1;
    for j = 2:N-1
        ii = i[j-1]>=i[j]+1>=i[j+1] ? j : ii
    end
    ni = i.+ntuple(x->x==ii ? 1 : 0,Val{N}());
    (SUNIrrep(i),ni)
end

Base.:(==)(s::SUNIrrep, t::SUNIrrep) = ==(s.I, t.I)
Base.hash(s::SUNIrrep, h::UInt) = hash(s.I, h)

TensorKit.dim(s::SUNIrrep) = dimension(Irrep(s.I))
Base.conj(s::SUNIrrep) = SUNIrrep((reverse(s.I) .- s.I[1]) .* -1) #maybe? https://aip.scitation.org/doi/pdf/10.1063/1.1704095
Base.one(::Type{SUNIrrep{N}}) where N = SUNIrrep(ntuple(n->0, N))

TensorKit.FusionStyle(::Type{SUNIrrep{N}}) where N = TensorKit.DegenerateNonAbelian()
Base.isreal(::Type{<:SUNIrrep}) = true
TensorKit.BraidingStyle(::Type{<:SUNIrrep}) = TensorKit.Bosonic();

TensorKit.:⊗(s1::SUNIrrep{N}, s2::SUNIrrep{N}) where N = (SUNIrrep(weight(k)) for k in keys(directproduct(Irrep(s1.I), Irrep(s2.I))))
TensorKit.Nsymbol(s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where N = get(directproduct(Irrep(s1.I), Irrep(s2.I)), Irrep(s3.I), 0)

TensorKit.fusiontensor(s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where N = CGC(Float64, Irrep(s1.I), Irrep(s2.I), Irrep(s3.I))

function TensorKit.Fsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}, d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where N
    N1 = Nsymbol(a,b,e)
    N2 = Nsymbol(e,c,d)
    N3 = Nsymbol(b,c,f)
    N4 = Nsymbol(a,f,d)

    (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0) && return fill(0.0, N1, N2, N3, N4)

    A = fusiontensor(a,b,e)
    B = fusiontensor(e,c,d)
    C = fusiontensor(b,c,f)
    D = fusiontensor(a,f,d)

    @tensor F[-1,-2,-3,-4] := conj(D[1,5,6,-4]) * conj(C[2,4,5,-3]) * A[1,2,3,-1] * B[3,4,6,-2]
    Array(F)/dim(d)
end

function TensorKit.Rsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where N
    N1 = Nsymbol(a, b, c);
    N2 = Nsymbol(b, a, c);

    (N1 == 0 || N2 == 0) && return fill(0.0, N1, N2)

    A = fusiontensor(a, b, c)
    B = fusiontensor(b, a, c)

    @tensor R[-1;-2] := conj(B[1, 2, 3, -2]) * A[2, 1, 3, -1]
    Array(R)/dim(c)
end
