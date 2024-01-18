const DISPLAY_MODES = ("weight", "dimension", "dynkin")
display_mode() = @load_preference("display_mode", "weight")
function display_mode(mode::AbstractString)
    mode in DISPLAY_MODES ||
        throw(ArgumentError("Invalid display mode $mode, needs to be one of $(DISPLAY_MODES)."))
    oldmode = display_mode()
    @set_preferences!("display_mode" => mode)
    return oldmode
end

function Base.show(io::IO, s::SUNIrrep)
    name = display_mode() == "weight" ? weightname(s) :
           display_mode() == "dynkin" ? dynkinname(s) :
           display_mode() == "dimension" ? "\"$(dimname(s))\"" :
           error("Invalid display mode $(display_mode()).")
    if get(io, :typeinfo, nothing) === typeof(s)
        print(io, name)
    else
        print(io, TensorKit.type_repr(typeof(s)), "(", name, ")")
    end
    return nothing
end

# Dynkin labels
# -------------

dynkinname(I::SUNIrrep) = "[$(join(dynkin_label(I), ", "))]"

"""
    dynkin_label(I::SUNIrrep)

Gives the labels of the Dynkin diagram of the SU(N) irrep `I` as a tuple of `N - 1`
integers. These are related to the Young Tableau by `aᵢ = λᵢ - λᵢ₊₁` where `λᵢ` is the
number of boxes in the `i`th row of the Young Tableau.
"""
dynkin_label(I::SUNIrrep{N}) where {N} = Z2weight(highest_weight(I))

# Weight names
# ------------

weightname(I::SUNIrrep) = string(weight(I))

# Dimensional names
# -----------------

max_dynkin_label(::Type{<:SUNIrrep}) = 5

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

Returns the index of the SU(N) irrep `I`.
"""
function index(s::SUNIrrep)
    N = s.N
    w = dynkin_label(s)
    metric = inverse_cartanmatrix(typeof(s))
    id = dim(s) * dot(collect(w), metric, collect(w) .+ 2) // (N^2 - 1)
    @assert denominator(id) == 1
    return numerator(id)
end

function all_dynkin(::Type{SUNIrrep{N}}, maxdynkin::Int=3) where {N}
    return (SUNIrrep(collect(I.I .- 1))
            for I in CartesianIndices(ntuple(k -> maxdynkin + 1, N - 1)))
end

function irreps_by_dim(::Type{SUNIrrep{N}}, d::Int, maxdynkin::Int=3) where {N}
    irreps = [I for I in all_dynkin(SUNIrrep{N}, maxdynkin) if dim(I) == d]
    return sort!(irreps; by=x -> (index(x), congruency(x), dynkin_label(x)))
end

function find_dimname(s::SUNIrrep{N}) where {N}
    a = dynkin_label(s)
    max_dynkin_label = N > 4 ? maximum(a) : maximum(a) + 1
    d = dim(s)

    same_dim_irreps = irreps_by_dim(typeof(s), d, max_dynkin_label)

    if length(same_dim_irreps) > 1
        ids = index.(same_dim_irreps)
        numprimes = findfirst(==(index(s)), unique!(ids)) - 1
        if congruency(s) == 0 || (iseven(N) && congruency(s) == N ÷ 2)
            conjugate = s != first(filter(x -> index(x) == index(s), same_dim_irreps))
        else
            conjugate = congruency(s) > congruency(dual(s))
        end
    elseif length(same_dim_irreps) == 1
        numprimes = 0
        conjugate = false
    else
        error("this should never happen")
    end

    return d, numprimes, conjugate
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

## References
- R. Slansky, “Group theory for unified model building,” Physics Reports, vol. 79, no. 1, pp. 1–128, Dec. 1981, doi: 10.1016/0370-1573(81)90092-2.
- R. Feger and T. W. Kephart, “LieART -- A Mathematica Application for Lie Algebras and Representation Theory,” arXiv, arXiv:1206.6379, Aug. 2014. doi: 10.48550/arXiv.1206.6379.
"""
function dimname(s::SUNIrrep{N}) where {N}
    # for some reason in SU{3}, the 6-dimensional irreps have switched duality
    s == SUNIrrep(2, 0, 0) && return generate_dimname(6, 0, false)
    s == SUNIrrep(2, 2, 0) && return generate_dimname(6, 0, true)

    d, numprimes, conjugate = find_dimname(s)
    return generate_dimname(d, numprimes, conjugate)
end

const str_dual = "†"
const str_prime = "′"

const prime_chars = ('\U2032', '\U2033', '\U2034', '\U2057')
const dual_char = '⁺'

function count_primes(name::Union{AbstractString,Vector{Char}})
    return sum(((i, c),) -> i * count(==(c), name), enumerate(prime_chars))
end
is_conjugate(name::AbstractString) = endswith(name, dual_char)

"""
    parse_dimname(name::AbstractString) -> (Int, Int, Bool)

Parses a dimensional name into a dimension, a number of primes and a conjugate flag.

See also: [`SUNRepresentations.generate_dimname`](@ref)
"""
function parse_dimname(name::AbstractString)
    name_chars = collect(name)
    # parse dimension part
    length(name_chars) > 0 || throw(ArgumentError("Cannot parse empty names."))

    n_digits′ = findfirst(!isdigit, name_chars)
    n_digits = isnothing(n_digits′) ? length(name_chars) : n_digits′ - 1

    n_digits > 0 || throw(ArgumentError("The name $name is not valid."))
    n_digits == length(name) || !any(isdigit, name_chars[(n_digits + 1):end]) ||
        throw(ArgumentError("The name $name is not valid."))

    d = parse(Int, name[1:n_digits])

    # parse conjugate part
    conjugate = is_conjugate(name)

    # parse prime part
    modifiers = name_chars[(n_digits + 1):(conjugate ? end - 1 : end)]
    isempty(modifiers) || all(in(prime_chars), modifiers) ||
        throw(ArgumentError("The name $name is not valid."))
    numprimes = count_primes(modifiers)

    return d, numprimes, conjugate
end

function add_primes(name::AbstractString, n::Int)
    n == 0 && return name
    name *= repeat(prime_chars[4], n ÷ 4)
    return n % 4 == 0 ? name : name * prime_chars[mod1(n, 4)]
end
add_conjugate(name::AbstractString, isconj=true) = isconj ? name * dual_char : name

"""
    generate_dimname(d::Int, numprimes::Int, conjugate::Bool) -> AbstractString

Generates a dimensional name from a dimension, a number of primes and a conjugate flag.

See also: [`SUNRepresentations.parse_dimname`](@ref)
"""
function generate_dimname(d::Int, numprimes::Int, conjugate::Bool)
    return add_conjugate(add_primes(string(d), numprimes), conjugate)
end
