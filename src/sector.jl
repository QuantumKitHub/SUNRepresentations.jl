# This is type piracy:
const SU₃ = SU{3}
const SU₄ = SU{4}
const SU₅ = SU{5}
TensorKitSectors.type_repr(::Type{SU₃}) = "SU₃"
TensorKitSectors.type_repr(::Type{SU₄}) = "SU₄"
TensorKitSectors.type_repr(::Type{SU₅}) = "SU₅"
Base.getindex(::TensorKitSectors.IrrepTable, ::Type{SU{N}}) where {N} = SUNIrrep{N, N - 1}

Base.IteratorSize(::Type{SectorValues{T}}) where {T <: SUNIrrep} = Base.IsInfinite()

Base.iterate(iter::SectorValues{<:SUNIrrep}, i::Int = 1) = iter[i], i + 1

# linear order of sectors: use manhattan ordering of dynkin labels (0-based)
function Base.getindex(::SectorValues{<:SUNIrrep{N}}, i::Int) where {N}
    sz = ntuple(Returns(TensorKitSectors.Unbounded()), N - 1)
    I = TensorKitSectors.manhattan_to_multidimensional_index(i, sz)
    dk_label = I .- 1 # dynkin labels are 0-based
    return SUNIrrep{N, N - 1}(map(UInt8, dk_label))  # direct inner constructor, no validation needed
end
function TensorKitSectors.findindex(::SectorValues{I}, s::I) where {I <: SUNIrrep}
    a = dynkin_label(s) .+ 1
    sz = ntuple(Returns(TensorKitSectors.Unbounded()), rank(I) - 1)
    return TensorKitSectors.to_manhattan_index(a, sz)
end

TensorKitSectors.dual(s::SUNIrrep) = typeof(s)(reverse(getfield(s, :a)))
TensorKitSectors.unit(::Type{I}) where {I <: SUNIrrep} = I(ntuple(Returns(zero(UInt8)), Val(rank(I) - 1)))

TensorKitSectors.FusionStyle(::Type{<:SUNIrrep}) = GenericFusion()
TensorKitSectors.BraidingStyle(::Type{<:SUNIrrep}) = Bosonic()

function TensorKitSectors.:⊗(s1::I, s2::I) where {I <: SUNIrrep}
    return SectorSet{I}(productirreps(s1, s2))
end
# Outer multiplicity of ν in λ ⊗ μ, as the number of Littlewood-Richardson tableaux that fill the
# skew diagram ν / μ with content λ.
function TensorKitSectors.Nsymbol(λ::I, μ::I, ν::I) where {N, I <: SUNIrrep{N}}
    # necessary conditions:
    # - the tableau adds |λ| boxes to μ, so |λ| + |μ| = |ν| + N d, where d ≥ 0 is the number of full
    #   columns that the normalisation of ν removed (equivalently, the congruency classes add up)
    d, r = divrem(nboxes(λ) + nboxes(μ) - nboxes(ν), N)
    (r == 0 && d >= 0) || return 0

    # - ν + d, the diagram with these columns restored, contains both λ and μ
    target = weight(ν) .+ d
    (all(map(<=, weight(λ), target)) && all(map(<=, weight(μ), target))) || return 0

    return littlewoodrichardson(Returns(nothing), λ, μ, target)
end
# Exact bounds used below, with n the number of boxes:
# - Dynkin labels are at most 255, so dim ≤ 256^(N(N - 1)/2) = 2^(4N(N - 1)) (all labels 255).
#   Hence dim fits in an Int for N ≤ 4.
# - dim ≤ Nⁿ (Schur-Weyl duality: the irrep appears in (ℂᴺ)^⊗n).
# - dim(s) = dim(dual(s)), and nboxes(dual(s)) = N * λ₁ - nboxes(s).
function TensorKitSectors.dim(s::SUNIrrep{N}) where {N}
    N <= 5 && return weyldim(s)
    # few boxes: the hook-content formula has n factors, the Weyl formula N(N - 1)/2
    n, n̄ = nboxes(s), nboxes(dual(s))
    min(n, n̄) < N * (N - 1) ÷ 2 || return weyldim(s)
    return n <= n̄ ? hookcontentdim(s, n) : hookcontentdim(dual(s), n̄)
end

# Hook-content formula: dim = ∏_{boxes (i, c)} (N + c - i) / hook(i, c). The numerator is at most
# (N + λ₁ - 1)ⁿ for n boxes, and the denominator divides it, which fixes the integer type up front.
function hookcontentdim(s::SUNIrrep{N}, n::Int = nboxes(s)) where {N}
    λ = weight(s)
    nbits = n * (64 - leading_zeros(N + λ[1] - 1))
    d = if nbits <= 62
        hookcontentdim(Int, λ)
    elseif nbits <= 126
        hookcontentdim(Int128, λ)
    else
        hookcontentdim(BigInt, λ)
    end
    d <= typemax(Int) || throw_dimoverflow(s)
    return Int(d)
end
function hookcontentdim(::Type{T}, λ::NTuple{N, Int}) where {T, N}
    num = den = one(T)
    L = something(findlast(>(0), λ), 0)
    for c in 1:λ[1]
        L = findprev(>=(c), λ, L)::Int # length of column c
        for i in 1:L
            num *= N + c - i
            den *= λ[i] - c + L - i + 1
        end
    end
    return div(num, den)
end

# Weyl dimension formula in integer arithmetic: the factors up to `k₂` give the dimension of an
# SU(k₂) irrep, so that the division after every row of factors is exact. The factors of row k₂
# are at most λ₁ + k₂ - 1, which bounds the product of that row; for N ≤ 6 it always fits in an
# Int128 (and in an Int for N ≤ 4), otherwise this is checked once per row.
function weyldim(s::SUNIrrep{N}) where {N}
    N > 34 && return bigweyldim(s) # (N - 1)! does not fit in an Int128
    T = N <= 4 ? Int : Int128
    I = weight(s)
    d = 1
    for k₂ in 2:N
        if N > 6
            nbits = (64 - leading_zeros(d)) + (k₂ - 1) * (64 - leading_zeros(I[1] + k₂ - 1))
            nbits <= 126 || return bigweyldim(s)
        end
        num, den = T(d), T(1)
        for k₁ in 1:(k₂ - 1)
            num *= k₂ - k₁ + I[k₁] - I[k₂]
            den *= k₂ - k₁
        end
        q, r = divrem(num, den)
        @assert r == 0
        N > 4 && q > typemax(Int) && throw_dimoverflow(s)
        d = Int(q)
    end
    return d
end
function bigweyldim(s::SUNIrrep{N}) where {N}
    I = weight(s)
    num = prod(big(k₂ - k₁ + I[k₁] - I[k₂]) for k₂ in 2:N for k₁ in 1:(k₂ - 1))
    den = prod(big(k₂ - k₁) for k₂ in 2:N for k₁ in 1:(k₂ - 1))
    d = div(num, den)
    d <= typemax(Int) || throw_dimoverflow(s)
    return Int(d)
end
@noinline throw_dimoverflow(s) = throw(OverflowError(lazy"dimension of $s does not fit in an Int"))

TensorKitSectors.sectorscalartype(::Type{<:SUNIrrep}) = Float64
TensorKitSectors.fusiontensor(s1::I, s2::I, s3::I) where {I <: SUNIrrep} =
    CGC(s1, s2, s3)

TensorKitSectors.fusionscalartype(::Type{<:SUNIrrep}) = Float64

TensorKitSectors.Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I <: SUNIrrep} =
    SUN_Fsymbol(a, b, c, d, e, f)
@cached function SUN_Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I)::Array{Float64, 4} where {I <: SUNIrrep}
    return TensorKitSectors.Fsymbol_from_fusiontensor(a, b, c, d, e, f)
end
MemoizationKit.default_cache_settings(::typeof(SUN_Fsymbol)) = (; maxsize = 10^5)
function TensorKitSectors.Fsymbol_from_fusiontensor(
        a::I, b::I, c::I, d::I, e::I, f::I
    ) where {I <: SUNIrrep}
    Nabe, Necd, Nbcd, Nafd = Nsymbol(a, b, e), Nsymbol(e, c, d), Nsymbol(b, c, f), Nsymbol(a, f, d)
    iszero(Nabe * Necd * Nbcd * Nafd) && return zeros(fusionscalartype(I), Nabe, Necd, Nbcd, Nafd)

    # computing first diagonal element
    A = fusiontensor(a, b, e)
    B = fusiontensor(e, c, d)[:, :, 1, :]
    C = fusiontensor(b, c, f)
    D = fusiontensor(a, f, d)[:, :, 1, :]

    @tensor F[-1, -2, -3, -4] := conj(D[1, 5, -4]) * conj(C[2, 4, 5, -3]) *
        A[1, 2, 3, -1] * B[3, 4, -2]
    return Array(F)
end

# remove the view calls
function TensorKitSectors.Asymbol_from_fusiontensor(a::I, b::I, c::I) where {I <: SUNIrrep}
    Nabc = Nsymbol(a, b, c)
    iszero(Nabc) && return zeros(fusionscalartype(typeof(a)), Nabc, Nabc)

    C1 = fusiontensor(a, b, c)[:, 1, :, :]
    C2 = fusiontensor(dual(a), c, b)[:, :, 1, :]
    Za = sqrtdim(a) * fusiontensor(a, dual(a), leftunit(a))[:, :, 1, 1]
    @tensor A[-1, -2] := sqrtdim(b) / sqrtdim(c) * conj(Za[1, 2]) * C1[1, 3, -1] * C2[2, 3, -2]
    return Array(A)
end
function TensorKitSectors.Bsymbol_from_fusiontensor(a::I, b::I, c::I) where {I <: SUNIrrep}
    Nabc = Nsymbol(a, b, c)
    iszero(Nabc) && return zeros(fusionscalartype(typeof(a)), Nabc, Nabc)

    C1 = fusiontensor(a, b, c)[1, :, :, :]
    C2 = fusiontensor(c, dual(b), a)[:, :, 1, :]
    Zb = sqrtdim(b) * fusiontensor(b, dual(b), leftunit(b))[:, :, 1, 1]
    @tensor B[-1, -2] := sqrtdim(a) / sqrtdim(c) * conj(Zb[1, 2]) * C1[1, 3, -1] * C2[3, 2, -2]
    return Array(B)
end

TensorKitSectors.braidingscalartype(::Type{<:SUNIrrep}) = Float64
TensorKitSectors.Rsymbol(a::I, b::I, c::I) where {I <: SUNIrrep} = SUN_Rsymbol(a, b, c)
@cached function SUN_Rsymbol(a::I, b::I, c::I)::Array{Float64, 2} where {I <: SUNIrrep}
    return TensorKitSectors.Rsymbol_from_fusiontensor(a, b, c)
end
MemoizationKit.default_cache_settings(::typeof(SUN_Rsymbol)) = (; maxsize = 10^5)
function TensorKitSectors.Rsymbol_from_fusiontensor(a::I, b::I, c::I) where {I <: SUNIrrep}
    Nabc = Nsymbol(a, b, c)
    iszero(Nabc) && return fill(zero(braidingscalartype(I)), Nabc, Nabc)

    A = fusiontensor(a, b, c)[:, :, 1, :]
    B = fusiontensor(b, a, c)[:, :, 1, :]

    @tensor R[-1; -2] := conj(B[1, 2, -2]) * A[2, 1, -1]
    return Array(R)
end
