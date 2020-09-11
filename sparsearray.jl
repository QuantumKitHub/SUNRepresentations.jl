using TensorOperations,SparseArrays
import LinearAlgebra;

struct SparseDOK{Elt,N} <: AbstractSparseArray{Elt,Int,N}
    eldict::Dict{NTuple{N,Int},Elt}
    sizes::NTuple{N,Int}
end

Base.eltype(d::SparseDOK{Elt}) where Elt = Elt
Base.size(d::SparseDOK) = d.sizes;
SparseDOK{Elt}(::UndefInitializer,sizes::Vararg{Int64,N}) where {Elt,N} = SparseDOK(Dict{NTuple{N,Int64},Elt}(),tuple(sizes...));
function Base.Array(d::SparseDOK{Elt,N}) where {Elt,N}
    f = zeros(Elt,d.sizes...);
    for (k,v) in d.eldict
        f[k...] = v
    end
    f
end
Base.getindex(d::SparseDOK{Elt,N},i::Vararg{Int64,N}) where {Elt,N} = get(d.eldict,tuple(i...),zero(Elt));
Base.setindex!(d::SparseDOK{Elt,N},v::Number,i::Vararg{Int64,N}) where {Elt,N} =  d.eldict[tuple(i...)] = v;

function LinearAlgebra.lmul!(a::Number, d::SparseDOK{Elt,N}) where {Elt,N}
    for k in keys(d.eldict)
        d.eldict[k]*=a;
    end
end
LinearAlgebra.rmul!(d::SparseDOK{Elt,N},a::Number) where {Elt,N} = LinearAlgebra.lmul!(a,d)

TensorOperations.similar_from_indices(T::Type, p1, p2, A::SparseDOK, CA::Symbol) = SparseDOK{T}(undef,[A.sizes...][p1...,p2...]...);
TensorOperations.similar_from_indices(T::Type, poA::Tuple{Vararg{Int64,N}} where N, poB::Tuple{Vararg{Int64,N}} where N,
                                p1::Tuple{Vararg{Int64,N}} where N, p2::Tuple{Vararg{Int64,N}} where N,
                                A::SparseDOK, B::SparseDOK, CA::Symbol, CB::Symbol) = SparseDOK{T}(undef,[[A.sizes...][poA...];[B.sizes...][poB...]][[p1...,p2...]]...);


function TensorOperations.add!(α, A::SparseDOK{Elt}, conjA::Symbol, β, C::SparseDOK{Elt}, indleft, indright) where Elt
    LinearAlgebra.lmul!(β,C);

    for (k,v) in A.eldict
        nk = tuple([k...][[indleft...,indright...]]...);

        C.eldict[nk] = get(C.eldict,nk,zero(Elt)) + α*(conjA==:C ? conj(v) : v);
    end
    C
end

function TensorOperations.trace!(α, A::SparseDOK, conjA::Symbol, β, C::SparseDOK,
    indleft, indright, cind1, cind2)
    throw(ArgumentError("not yet implement"))
end

function TensorOperations.contract!(α, A::SparseDOK{Elt} where N, conjA::Symbol, B::SparseDOK{Elt}, conjB::Symbol, β, C::SparseDOK{Elt},
    oindA::Tuple{Vararg{Int}}, cindA::Tuple{Vararg{Int}}, oindB::Tuple{Vararg{Int}}, cindB::Tuple{Vararg{Int}},
    indleft::Tuple{Vararg{Int}}, indright::Tuple{Vararg{Int}}, syms) where Elt
    LinearAlgebra.lmul!(β,C);

    for (kA,vA) in A.eldict

        for (kB,vB) in B.eldict
            [kB[i] for i in cindB] == [kA[i] for i in cindA] || continue;


            kC = tuple([[kA[i] for i in oindA];[kB[i] for i in oindB]][[indleft...,indright...]]...)
            C.eldict[kC] = get(C.eldict,kC,zero(Elt)) + α*(conjA == :C ? conj(vA) : vA)*(conjB == :C ? conj(vB) : vB)
        end
    end
    C
end

A = SparseDOK{Float64}(undef,2,2,3);
A[1,1,1] = 1;
@tensor v[-1;-2]:=A[1,2,-2]*A[1,2,-1]

fA = Array(A);
@tensor fv[-1;-2]:=fA[1,2,-2]*fA[1,2,-1]
@show LinearAlgebra.norm(collect(v)-fv)
