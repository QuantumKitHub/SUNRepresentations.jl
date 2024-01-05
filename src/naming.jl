"""
    dynkin_label(I::SUNIrrep)

Gives the labels of the Dynkin diagram of the SU(N) irrep `I` as a tuple of `N - 1`
integers. These are related to the Young Tableau by `aᵢ = λᵢ - λᵢ₊₁` where `λᵢ` is the
number of boxes in the `i`th row of the Young Tableau.
"""
dynkin_label(I::SUNIrrep{N}) where {N} = Z2weight(highest_weight(I))

function from_dynkin_label(a::NTuple{N,Int}) where {N}
    w = reverse(cumsum(reverse(a)))
    return SUNIrrep(w..., 0)
end

max_dynkin_label(::Type{<:SUNIrrep}) = 3

"""
    congruency(I::SUNIrrep)

Returns the congruency class of the SU(N) irrep `I`, which expresses to what class of the ℤₙ-grading the irrep belongs.
"""
function congruency(I::SUNIrrep{N}) where {N}
    return sum(((k, aₖ),) -> aₖ * k, enumerate(dynkin_label(I))) % N
end

cartanmatrix(I::SUNIrrep) = cartanmatrix(typeof(I))
function cartanmatrix(::Type{SUNIrrep{N}}) where {N}
    A = zeros(Int, N - 1, N - 1)
    for i in 1:(N - 1), j in 1:(N - 1)
        A[i, j] = 2 * (i == j) - (i == j + 1) - (i == j - 1)
    end
    return A
end
inverse_cartanmatrix(I::SUNIrrep) = inverse_cartanmatrix(typeof(I))
function inverse_cartanmatrix(::Type{SUNIrrep{N}}) where {N}
    A⁻¹ = zeros(Int, N - 1, N - 1)
    for i in 1:(N - 1), j in i:(N - 1)
        A⁻¹[i, j] = i * (N - j)
        A⁻¹[j, i] = A⁻¹[i, j]
    end
    return A⁻¹ .// N
end

"""
    index(I::SUNIrrep)

Returns the index of the SU(N) irrep `I`
"""
function index(s::SUNIrrep)
    N = s.N
    w = dynkin_label(s)
    metric = inverse_cartanmatrix(typeof(s))
    id = dim(s) * dot(collect(w), metric, collect(w) .+ 2) // (N^2 - 1)
    @assert denominator(id) == 1
    return numerator(id)
end

function irreps_by_dim(::Type{SUNIrrep{N}}, d::Int, maxdynkin::Int=3) where {N}
    irreps = SUNIrrep{N}[]
    
    all_dynkin = CartesianIndices(ntuple(k -> maxdynkin + 1, N-1))
    for a in all_dynkin
        I = from_dynkin_label(a.I .- 1)
        dim(I) == d && push!(irreps, I)
    end
    
    return sort!(irreps; by=x -> (index(x), congruency(x), dynkin_label(x)...))
end

"""
    dimname(I::SUNIrrep) -> String

Returns the dimensional name of an irrep, e.g. "6" for the 6-dimensional irrep of SU(3).
When there are multiple irreps with the same dimension, they are sorted by index and the
number of primes `′` indicates the position in that list.
Dual representations have the same
dimension and index, and by convention the one with the lowest congruency class is chosen as
the "non-dual" one, and the other one is marked with a `†`.
If the congruency class are the
same, the one with the lowest Dynkin label, compared lexicographically, is chosen as the
"non-dual" one.

!!! warning

    This function necessarily has to scan through all irreps to list those that have the
    same dimension, and therefore an appropriate cutoff has to be chosen. By default, the
    search space is all irreps with Dynkin labels up to 1 higher than the maximal label of
    `s` for SU{N} with `N <= 4`, and up to the maximal Dynkin label for `N > 4`. This means
    that it is possible that the number of primes is not consistent. Nevertheless, these
    labels are never used internally and should thus not cause any problems.
"""
function dimname(s::SUNIrrep{N}) where {N}
    # for some reason in SU{3}, the 6-dimensional irreps have switched duality
    s == SUNIrrep(2, 0, 0) && return "6"
    s == SUNIrrep(2, 2, 0) && return "6†"
    
    a = dynkin_label(s)
    max_dynkin_label = N > 4 ? maximum(a) : maximum(a) + 1
    d = dim(s)
    
    same_dim_irreps = irreps_by_dim(typeof(s), d, max_dynkin_label)
    
    if length(same_dim_irreps) > 1
        ids = index.(same_dim_irreps)
        numprimes = findfirst(==(index(s)), unique!(ids)) - 1
        if congruency(s) == 0
            conjugate = s != first(filter(x -> index(x)==(index(s)), same_dim_irreps))
        else
            conjugate = congruency(s) > congruency(dual(s))
        end
    else
        numprimes = 0
        conjugate = false
    end

    name = conjugate ? string(d) * str_dual : string(d)
    return name * repeat(str_prime, numprimes)
end

const str_dual = "†"
const str_prime = "′"

function SUNIrrep{N}(name::AbstractString) where {N}
    d = parse(Int, filter(isdigit, name))
    numprimes = count(x -> x == str_prime, name)
    conjugate = contains(name, str_dual)

    max_dynkin = max_dynkin_label(SUNIrrep{N})

    same_dim_irreps = irreps_by_dim(SUNIrrep{N}, d, max_dynkin)
    isempty(same_dim_irreps) &&
        throw(ArgumentError("Either the name $name is not valid for SU{$N} or the irrep has at least one Dynkin label higher than $max_dynkin. You can expand the search space with `SUNRepresentations.max_dynkin_label(SUNIrrep{$N}) = a`."))

    id = unique!(index.(same_dim_irreps))[numprimes + 1]
    same_id_irreps = filter(x -> index(x) == id, same_dim_irreps)

    return conjugate ? last(same_id_irreps) : first(same_id_irreps)
end
