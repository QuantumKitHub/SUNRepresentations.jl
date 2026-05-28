"""
    struct SUNIrrep{N, M} <: AbstractIrrep{SU{N}}

Irrep of SU(N) labelled by its Dynkin labels `a = (a₁, …, a_{N-1})`, stored as `NTuple{M, UInt8}` where `M = N - 1`.
The Dynkin labels are related to the highest weight `λ` by `aᵢ = λᵢ − λᵢ₊₁`.

# Constructors

The various constructors reflect the different conventions for specifying ``SU(N)`` representations,
through weights, Dynkin labels or dimensional names.
To uniquely identify the target irrep, the value of `N` must always be supplied.

    SUNIrrep{N}(weight::NTuple{N, Int})

Constructs from the N-component highest weight
(shift-invariant: any representative is accepted; the stored Dynkin labels are the canonical form).

    SUNIrrep{N}(dynkin_label::NTuple{N - 1, Int})

Constructs directly from the `N - 1` Dynkin labels.

    SUNIrrep{N}(args::Vararg{Int})

Vararg form: `N` arguments are interpreted as weight components, `N - 1` as Dynkin labels.

    SUNIrrep{N}(name::AbstractString)

Constructs from a dimensional name such as `"8"` or `"6'"`.

See also: [`weight`](@ref), [`dynkin_label`](@ref).
"""
struct SUNIrrep{N, M} <: AbstractIrrep{SU{N}}
    a::NTuple{M, UInt8}
    function SUNIrrep{N, M}(a::NTuple{M, UInt8}) where {N, M}
        M == N - 1 || _throw_typeerror(N, M)
        return new{N, M}(a)
    end
end

@noinline _throw_typeerror(N, M) = throw(TypeError(:SUNIrrep, SUNIrrep{N, N - 1}, SUNIrrep{N, M}))
# --- NTuple constructors (primary implementations in two-parameter form) ---
# Conversion between weight and Dynkin labels:
#
# Highest weight:  λ = (λ₁ ≥ λ₂ ≥ … ≥ λₙ),  λₙ = 0  (normalised)
# Dynkin labels:   aᵢ = λᵢ − λᵢ₊₁  (shift-invariant, always ≥ 0)

_weight_to_dynkin(w::NTuple{N, Integer}) where {N} =
    ntuple(i -> Int(w[i]) - Int(w[i + 1]), Val(N - 1))

_dynkin_to_weight(a::NTuple{M, Integer}) where {M} =
    M == 0 ? (0,) : ((_dynkin_to_weight(Base.front(a)) .+ last(a))..., 0)

function SUNIrrep{N, M}(t::NTuple{O, Integer}) where {N, M, O}
    M == N - 1 || _throw_typeerror(N, M)

    if O == N # Weight constructor: N components → SU(N). Normalises automatically.
        d = _weight_to_dynkin(t)
        return SUNIrrep{N, M}(d)
    end

    O == M || throw(ArgumentError(lazy"SUNIrrep{$N, $M} requires either a weight ($N integers), or Dynkin labels ($M integers), got $O."))

    return SUNIrrep{N, M}(map(UInt8, t))
end

# Vararg shim: slurps into NTuple constructors (both weight and Dynkin paths).
SUNIrrep{N, M}(args::Vararg{Int}) where {N, M} = SUNIrrep{N, M}(args)

# One-parameter forms: thin delegates to the two-parameter constructors.
SUNIrrep{N}(args...) where {N} = SUNIrrep{N, N - 1}(args...)

# Helpful errors when {N} is omitted.
SUNIrrep(args::Vararg{Int}) = throw(
    ArgumentError(
        "SUNIrrep requires an explicit group rank. " *
            "Use SUNIrrep{N}(args...) where N is the SU(N) rank, " *
            "passing either N weight components or N-1 Dynkin labels."
    )
)
SUNIrrep(::Tuple) = throw(
    ArgumentError(
        "SUNIrrep requires an explicit group rank. Use SUNIrrep{N}(t)."
    )
)

function SUNIrrep{N, M}(name::AbstractString) where {N, M}
    M == N - 1 || throw(
        ArgumentError(
            "SUNIrrep{$N,$M}: second type parameter must equal N-1 = $(N - 1)"
        )
    )
    if N == 3
        name == generate_dimname(6, 0, false) && return SUNIrrep{N, M}(2, 0, 0)
        name == generate_dimname(6, 0, true) && return SUNIrrep{N, M}(2, 2, 0)
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

Base.convert(::Type{<:SUNIrrep{N}}, I::NTuple{N, Int}) where {N} = SUNIrrep{N}(I)       # weight (N components)
Base.convert(::Type{<:SUNIrrep{N}}, I::NTuple{M, Int}) where {N, M} = SUNIrrep{N}(I)  # Dynkin (N-1 components)
Base.convert(::Type{<:SUNIrrep{N}}, I::AbstractString) where {N} = SUNIrrep{N}(I)

const SU3Irrep = SUNIrrep{3, 2}
const SU4Irrep = SUNIrrep{4, 3}
const SU5Irrep = SUNIrrep{5, 4}

# --- Accessors ---
"""
    dynkin_label(I::SUNIrrep)

Gives the labels of the Dynkin diagram of the SU(N) irrep `I` as a tuple of `N - 1` integers.
These are related to the Young Tableau by `aᵢ = λᵢ - λᵢ₊₁` where `λᵢ` is the number of boxes in the `i`th row of the Young Tableau.

See also: [`weight`](@ref).
"""
dynkin_label(s::SUNIrrep) = map(Int, getfield(s, :a))

"""
    weight(s::SUNIrrep{N}) -> NTuple{N, Int}

Return the highest weight of the SU(N) irrep `s` as a normalised `N`-tuple of integers (last component = 0).
The components satisfy `λ₁ ≥ λ₂ ≥ … ≥ λₙ = 0`.

This is derived from the stored Dynkin labels via suffix sums.

See also: [`dynkin_label`](@ref).
"""
weight(s::SUNIrrep) = _dynkin_to_weight(dynkin_label(s))

"""
    rank(s::SUNIrrep) -> Int
    rank(::Type{<:SUNIrrep{N}}) -> Int

Return the rank `N` of the SU(N) group associated with the irrep `s`.

See also: [`SUNIrrep`](@ref).
"""
rank(s::SUNIrrep) = rank(typeof(s))
rank(::Type{<:SUNIrrep{N}}) where {N} = N

function Base.getproperty(s::SUNIrrep{N}, f::Symbol) where {N}
    f == :N && return rank(s)
    f == :a && return dynkin_label(s)
    f == :I && return weight(s)
    return getfield(s, f)
end

# --- Derived properties ---
"""
    casimir(k::Int, irrep::SUNIrrep)

Return the eigenvalue of the `k`-th order Casimir operator in the representation `irrep`.

The formula is:
```math
C_k(\\lambda) = \\frac{1}{2} \\left[ \\sum_{i=1}^{N} L_i^k - \\sum_{i=1}^{N} \\rho_i^k \\right]
```
where ``\\rho_i = (N+1-2i)/2`` are the Weyl vector components and
``L_i = (\\lambda_i - \\bar\\lambda) + \\rho_i`` are the shifted traceless weights
(``\\bar\\lambda = \\sum_j \\lambda_j / N``).

The independent primitive Casimir operators have orders ``k = 2, 3, \\ldots, N``.
Other values of `k` are valid but give dependent (or zero) results.
"""
function casimir(k::Int, irrep::SUNIrrep{N}) where {N}
    λ = weight(irrep)
    λ̄ = sum(λ) // N
    c = zero(Rational{Int})
    for i in 1:N
        ρᵢ = (N + 1 - 2i) // 2
        Lᵢ = (λ[i] - λ̄) + ρᵢ
        c += Lᵢ^k - ρᵢ^k
    end
    return c / 2
end

"""
    congruency(I::SUNIrrep)

Returns the congruency class of the SU(N) irrep `I`, which expresses to what class of the ℤₙ-grading the irrep belongs.
"""
function congruency(I::SUNIrrep{N}) where {N}
    return sum(((k, aₖ),) -> aₖ * k, enumerate(dynkin_label(I))) % N
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

cartanmatrix(I::SUNIrrep) = cartanmatrix(typeof(I))
function cartanmatrix(::Type{<:SUNIrrep{N}}) where {N}
    A = Matrix{Int}(undef, N - 1, N - 1)
    @inbounds for I in eachindex(IndexCartesian(), A)
        i, j = Tuple(I)
        A[I] = 2 * (i == j) - (i == j + 1) - (i == j - 1)
    end
    return A
end

inverse_cartanmatrix(I::SUNIrrep) = inverse_cartanmatrix(typeof(I))
function inverse_cartanmatrix(::Type{<:SUNIrrep{N}}) where {N}
    A⁻¹ = Matrix{Int}(undef, N - 1, N - 1)
    @inbounds for I in eachindex(IndexCartesian(), A⁻¹)
        i, j = minmax(Tuple(I)...)
        A⁻¹[I] = i * (N - j)
    end
    return A⁻¹ .// N
end

# --- Equality and comparison ---
Base.hash(s::SUNIrrep, h::UInt) = hash(getfield(s, :a), h)
Base.:(==)(s::SUNIrrep, t::SUNIrrep) = getfield(s, :a) == getfield(t, :a)

function Base.isless(s1::I, s2::I) where {I <: SUNIrrep}
    I1 = dynkin_label(s1)
    I2 = dynkin_label(s2)
    d1 = sum(I1)
    d2 = sum(I2)
    d1 < d2 && return true
    d1 > d2 && return false
    return isless(I1, I2)
end
