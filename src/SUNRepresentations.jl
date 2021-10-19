module SUNRepresentations

using TensorOperations
using SparseArrayKit
using RationalRoots
using LinearAlgebra
using TensorKit;
using TensorKit: fusiontensor, Nsymbol
using LRUCache

export SUNIrrep, basis, weight, Zweight, creation, annihilation, highest_weight,dim
export directproduct, CGC

struct SUNIrrep{N} <: TensorKit.AbstractIrrep{TensorKit.SU{N}}
    I::NTuple{N,Int}
end
SUNIrrep(args::Vararg{Int}) = SUNIrrep(args)
Base.isless(s1::SUNIrrep{N}, s2::SUNIrrep{N}) where N = isless(s1.I, s2.I)

_normalize(s::SUNIrrep) = (I = weight(s); return SUNIrrep(I .- I[end]))

Base.getproperty(s::SUNIrrep{N}, f::Symbol) where {N} = f == :N ? N : getfield(s, f)
weight(s::SUNIrrep) = getfield(s, :I)

function TensorKit.dim(s::SUNIrrep)
    N = s.N
    I = weight(s)
    dim = 1//1
    for k2 = 2:N, k1 = 1:k2-1
        dim *= (k2 - k1 + I[k1] - I[k2]) // (k2 - k1)
    end
    @assert denominator(dim) == 1
    return numerator(dim)
end

include("gtpatterns.jl")

basis(s::SUNIrrep) = GTPatternIterator(s)

# direct product: return dictionary with new irreps as keys, outer multiplicities as value
function directproduct(s1::SUNIrrep{N}, s2::SUNIrrep{N}) where {N}
    dim(s1) > dim(s2) && return directproduct(s2, s1)
    result = Dict{SUNIrrep{N}, Int}()
    for m in basis(s1)
        t = weight(s2)
        bad_pattern = false
        for k in 1:N, l=N:-1:k
            bad_pattern && break
            bkl = m[k,l]
            if checkbounds(m, k, l-1)
                bkl -= m[k,l-1]
            end
            t = Base.setindex(t, t[l] + bkl, l)
            if l>1
                bad_pattern = t[l-1]< t[l]
            end
        end
        if !(bad_pattern)
            s = _normalize(SUNIrrep(t))
            result[s] = get(result, s, 0) + 1
        end
    end
    return result
end

include("clebschgordan.jl")
include("sector.jl")

end
