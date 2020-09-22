module SUNRepresentations

using TensorOperations
using TensorOperations: SparseArray, nonzeros
using RationalRoots
using Requires
using LinearAlgebra

export Irrep, dimension, basis, weight, creation, annihilation, highest_weight
export directproduct
export GTPattern, GTPatternIterator, Zweight, Z2weight, Z2weightmap

struct Irrep{N} # Irrep of SU(N)
    I::NTuple{N,Int}
end
Irrep(args::Vararg{Int}) = Irrep(args)
Base.isless(s1::Irrep{N}, s2::Irrep{N}) where N = isless(s1.I, s2.I)

_normalize(s::Irrep) = (I = weight(s); return Irrep(I .- I[end]))

Base.getproperty(s::Irrep{N}, f::Symbol) where {N} = f == :N ? N : getfield(s, f)
weight(s::Irrep) = getfield(s, :I)

function dimension(s::Irrep)
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

basis(s::Irrep) = GTPatternIterator(s)

# direct product: return dictionary with new irreps as keys, outer multiplicities as value
function directproduct(s1::Irrep{N}, s2::Irrep{N}) where {N}
    dimension(s1) > dimension(s2) && return directproduct(s2, s1)
    result = Dict{Irrep{N}, Int}()
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
            s = _normalize(Irrep(t))
            result[s] = get(result, s, 0) + 1
        end
    end
    return result
end

include("clebschgordan.jl")


function __init__()
    @require TensorKit="07d1fe3e-3e46-537d-9eac-e9e13d0d4cec" begin
        include("sector.jl")
    end
end


end
