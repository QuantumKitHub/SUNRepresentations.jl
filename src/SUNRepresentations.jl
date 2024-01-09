module SUNRepresentations

using TensorOperations
using SparseArrayKit
using RationalRoots
using LinearAlgebra
using TensorKit
using TensorKit: fusiontensor, Nsymbol, SU
using LRUCache
using Preferences

export SUNIrrep, basis, weight, Zweight, creation, annihilation, highest_weight, dim
export directproduct, CGC
export SU, SU₃, SU₄, SU₅, SU3Irrep, SU4Irrep, SU5Irrep
export dynkin_label, congruency

"""
    struct SUNIrrep{N} <: TensorKit.AbstractIrrep{SU{N}}

The irrep of SU(N) with highest weight `I`.

# Constructors

    SUNIrrep(I::NTuple{N,Int})
    SUNIrrep(args::Vararg{Int})

Constructs the `SU{N}` irrep with highest weight `I` or `args...`.

    SUNIrrep{N}(name::AbstractString)

Constructs the `SU{N}` irrep with dimensional name `name`. Note that the parameter `N` is
required to uniquely identify the irrep.

    SUNIrrep(a::Vector{Int})

Constructs the `SU{N}` irrep with highest weight `a = [a₁, a₂, …, aₙ₋₁]`.
"""
struct SUNIrrep{N} <: TensorKit.AbstractIrrep{SU{N}}
    I::NTuple{N,Int}
end

SUNIrrep(args::Vararg{Int,N}) where {N} = SUNIrrep{N}(args)
SUNIrrep{N}(args::Vararg{Int}) where {N} = SUNIrrep{N}(args)

SUNIrrep(a::Vector{Int}) = SUNIrrep{length(a)+1}(a)
function SUNIrrep{N}(a::Vector{Int}) where {N}
    @assert length(a) == N - 1
    return SUNIrrep{N}(reverse(cumsum(reverse(a)))..., 0)
end

function SUNIrrep{N}(name::AbstractString) where {N}
    if N == 3
        name == generate_dimname(6, 0, false) && return SUNIrrep{N}(2, 0, 0)
        name == generate_dimname(6, 0, true) && return SUNIrrep{N}(2, 2, 0)
    end
    
    d, numprimes, conjugate = parse_dimname(name)
    max_dynkin = max_dynkin_label(SUNIrrep{N})

    same_dim_irreps = irreps_by_dim(SUNIrrep{N}, d, max_dynkin)
    same_dim_ids = unique!(index.(same_dim_irreps))
    length(same_dim_ids) < numprimes + 1 &&
        throw(ArgumentError("Either the name $name is not valid for SU{$N} or the irrep has at least one Dynkin label higher than $max_dynkin.\nYou can expand the search space with `SUNRepresentations.max_dynkin_label(SUNIrrep{$N}) = a`."))
    
    id = same_dim_ids[numprimes + 1]
    same_id_irreps = filter(x -> index(x) == id, same_dim_irreps)
    @assert length(same_id_irreps) <= 2
    return conjugate ? last(same_id_irreps) : first(same_id_irreps)
end

const SU3Irrep = SUNIrrep{3}
const SU4Irrep = SUNIrrep{4}
const SU5Irrep = SUNIrrep{5}

Base.isless(s1::SUNIrrep{N}, s2::SUNIrrep{N}) where {N} = isless(s1.I, s2.I)

_normalize(s::SUNIrrep) = (I = weight(s); return SUNIrrep(I .- I[end]))

Base.getproperty(s::SUNIrrep{N}, f::Symbol) where {N} = f == :N ? N : getfield(s, f)
weight(s::SUNIrrep) = getfield(s, :I)

function TensorKit.dim(s::SUNIrrep)
    N = s.N
    I = weight(s)
    dim = 1 // 1
    for k2 in 2:N, k1 in 1:(k2 - 1)
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
    result = Dict{SUNIrrep{N},Int}()
    for m in basis(s1)
        t = weight(s2)
        bad_pattern = false
        for k in 1:N, l in N:-1:k
            bad_pattern && break
            bkl = m[k, l]
            if checkbounds(m, k, l - 1)
                bkl -= m[k, l - 1]
            end
            t = Base.setindex(t, t[l] + bkl, l)
            if l > 1
                bad_pattern = t[l - 1] < t[l]
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
include("naming.jl")

end
