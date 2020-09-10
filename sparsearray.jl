using TensorOperations
import LinearAlgebra;

#=
Couldn't find this silly thing anywhere
It's not fast, it's not fancy, but it works ...
=#
struct SparseDOK{Elt,N}# <: AbstractArray{Elt,N}
    eldict::Dict{NTuple{N,Int64},Elt}
    sizes::NTuple{N,Int64}
end

Base.eltype(d::SparseDOK{Elt}) where Elt = Elt
Base.size(d::SparseDOK) = d.sizes;
SparseDOK{Elt}(::UndefInitializer,sizes::Vararg{Int64,N}) where {Elt,N} = SparseDOK(Dict{NTuple{N,Int64},Elt}(),tuple(sizes...));

function LinearAlgebra.lmul!(a, d::SparseDOK{Elt,N}) where {Elt,N}
    for k in keys(d.eldict)
        d.eldict[k]*=a;
    end
end
LinearAlgebra.rmul!(d::SparseDOK{Elt,N},a) where {Elt,N} = LinearAlgebra.lmul!(a,d)


TensorOperations.similar_from_indices(T::Type, p1, p2, A::SparseDOK, CA::Symbol) = SparseDOK{T}(undef,[A.sizes...][p1...,p2...]...);
TensorOperations.similar_from_indices(T::Type, poA::Tuple{Vararg{Int64,N}} where N, poB::Tuple{Vararg{Int64,N}} where N,
                                p1::Tuple{Vararg{Int64,N}} where N, p2::Tuple{Vararg{Int64,N}} where N,
                                A::SparseDOK, B::SparseDOK, CA::Symbol, CB::Symbol) = SparseDOK{T}(undef,[[A.sizes...][poA...];[B.sizes...][poB...]][[p1...,p2...]]...);

#=
Implements C = β*C+α*permute(op(A)) where A is permuted such that the left (right)
 indices of C correspond to the indices indleft (indright) of A,
 and op is conj if conjA == :C or the identity map if conjA == :N (default).
 Together, (indleft..., indright...) is a permutation of 1 to the number of indices (dimensions) of A.
=#
function TensorOperations.add!(α, A::SparseDOK{Elt}, conjA::Symbol, β, C::SparseDOK{Elt}, indleft, indright) where Elt
    LinearAlgebra.lmul!(β,C);

    for (k,v) in A.eldict
        nk = tuple([k...][[indleft...,indright...]]...);

        C.eldict[nk] = get(C.eldict,nk,zero(Elt)) + α*(conjA==:C ? conj(v) : v);
    end
    C
end

#=
Implements C = β*C+α*partialtrace(op(A)) where A is permuted and partially traced,
such that the left (right) indices of C correspond to the indices indleft (indright) of A,
and indices cindA1 are contracted with indices cindA2. Furthermore, op is conj if conjA == :C
or the identity map if conjA=:N (default). Together, (indleft..., indright..., cind1, cind2)
is a permutation of 1 to the number of indices (dimensions) of A.
=#
function TensorOperations.trace!(α, A::SparseDOK, conjA::Symbol, β, C::SparseDOK,
    indleft, indright, cind1, cind2)
    throw(ArgumentError("not yet implement"))
end

#=
Implements C = β*C+α*contract(opA(A),opB(B)) where A and B are contracted,
such that the indices cindA of A are contracted with indices cindB of B.
The open indices oindA of A and oindB of B are permuted such that C has left (right) indices
corresponding to indices indleft (indright) out of (oindA..., oindB...).
The operation opA (opB) acts as conj if conjA (conjB) equal :C or
as the identity map if conjA (conjB) equal :N. Together, (oindA..., cindA...) is
a permutation of 1 to the number of indices of A and (oindB..., cindB...) is a permutation
of 1 to the number of indices of C. Furthermore, length(cindA) == length(cindB),
length(oindA)+length(oindB) equals the number of indices of C and (indleft..., indright...) i
s a permutation of 1 ot the number of indices of C.

The final argument syms is optional and can be either nothing,
or a tuple of three symbols,
which are used to identify temporary objects in the cache to be used for
permuting A, B and C so as to perform the contraction as a matrix multiplication.
=#

function TensorOperations.contract!(α, A::SparseDOK{Elt}, conjA::Symbol, B::SparseDOK, conjB::Symbol, β, C::SparseDOK,
    oindA, cindA, oindB, cindB,
    indleft, indright, syms) where Elt
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
A.eldict[(1,1,1)] = 1;
@tensor v[-1;-2]:=A[1,2,-2]*A[1,2,-1]
@show v.eldict
