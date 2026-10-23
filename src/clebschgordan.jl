const TOL_NULLSPACE = 1.0e-13
# tolerance for nullspace
const TOL_GAUGE = 1.0e-11
# tolerance for gaugefixing should probably be bigger than that with which nullspace was determined
const TOL_PURGE = 1.0e-14
# tolerance for dropping zeros

function weightmap(basis)
    N = first(basis).N
    # basis could be a GTPatternIterator{N}, but also a Vector{GTPattern{N}}
    weights = Dict{NTuple{N, Int}, Vector{Int}}()
    for (i, m) in enumerate(basis)
        w = weight(m)
        push!(get!(weights, w, Int[]), i)
    end
    return weights
end

CGC(s1::I, s2::I, s3::I) where {I <: SUNIrrep} = CGC(sectorscalartype(I), s1, s2, s3)
CGC(::Type{T}, s1::I, s2::I, s3::I) where {T, I <: SUNIrrep} = SUN_CGC(T, s1, s2, s3)

@cached function SUN_CGC(::Type{T}, s1::I, s2::I, s3::I)::SparseArray{T, 4} where {T <: Real, I <: SUNIrrep}
    if isone(s1)
        @assert s2 == s3
        CGC = trivial_CGC(T, s2, true)
    elseif isone(s2)
        @assert s1 == s3
        CGC = trivial_CGC(T, s1, false)
    else
        CGC = highest_weight_CGC(T, s1, s2, s3)
        lower_weight_CGC!(CGC, s1, s2, s3)
        purge!(CGC)
    end
    @debug "Computed CGC: $s1 ⊗ $s2 → $s3"
    return CGC
end

# The Clebsch-Gordan coefficients are cached in RAM and on disk.
MemoizationKit.DiskCacheStyle(::typeof(SUN_CGC), args...) = DiskCache(; serializer = CGCSerializer)
MemoizationKit.diskversion(::typeof(SUN_CGC)) = "1"
MemoizationKit.default_cache_settings(::typeof(SUN_CGC)) = (; maxsize = 10^5)

# Serializer of the disk cache, which stores the coefficients in a compact format that does not
# depend on the Julia version: the size, and the linear indices and values of the nonzero entries.
# Everything else, such as the keys, is written as by `Serialization`.
mutable struct CGCSerializer{I <: IO} <: Serialization.AbstractSerializer
    io::I
    counter::Int
    table::IdDict{Any, Any}
    pending_refs::Vector{Int}
    version::Int
    CGCSerializer(io::I) where {I <: IO} = new{I}(io, 0, IdDict(), Int[], 0)
end

function Serialization.serialize(s::CGCSerializer, A::SparseArray{T, 4}) where {T <: Base.IEEEFloat}
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, typeof(A))
    entries = collect(nonzero_pairs(A))
    linear = LinearIndices(A)
    write(s.io, collect(Int64, size(A)), Int64(length(entries)))
    write(s.io, Int64[linear[first(e)] for e in entries], T[last(e) for e in entries])
    return nothing
end
function Serialization.deserialize(s::CGCSerializer, ::Type{SparseArray{T, 4}}) where {T <: Base.IEEEFloat}
    dims = NTuple{4, Int}(read!(s.io, Vector{Int64}(undef, 4)))
    n = read(s.io, Int64)
    linear = read!(s.io, Vector{Int64}(undef, n))
    values = read!(s.io, Vector{T}(undef, n))
    cartesian = CartesianIndices(dims)
    data = Dict{CartesianIndex{4}, T}(cartesian[i] => v for (i, v) in zip(linear, values))
    return SparseArray{T, 4}(data, dims)
end

gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))

# special case for 1 ⊗ s -> s or s ⊗ 1 -> s
function trivial_CGC(::Type{T}, s::SUNIrrep, isleft = true) where {T <: Real}
    d = dim(s)
    if isleft
        CGC = SparseArray{T}(undef, 1, d, d, 1)
        for m in 1:d
            CGC[1, m, m, 1] = one(T)
        end
    else
        CGC = SparseArray{T}(undef, d, 1, d, 1)
        for m in 1:d
            CGC[m, 1, m, 1] = one(T)
        end
    end
    return CGC
end

const _emptyindexlist = Vector{Int}()

function highest_weight_CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I <: SUNIrrep}
    d1, d2, d3 = dim(s1), dim(s2), dim(s3)
    N = s1.N

    Jp_list1 = creation(s1)
    Jp_list2 = creation(s2)
    eqs = SparseArray{T}(undef, N - 1, d1, d2, d1, d2)

    cols = Vector{CartesianIndex{2}}()
    rows = Vector{CartesianIndex{3}}()

    map2 = weightmap(basis(s2))
    w3 = weight(highest_weight(s3))
    wshift = div(nboxes(s1) + nboxes(s2) - nboxes(s3), N)

    for (m1, pat1) in enumerate(basis(s1))
        w1 = weight(pat1)
        w2 = w3 .- w1 .+ wshift
        for m2 in get(map2, w2, _emptyindexlist)
            push!(cols, CartesianIndex(m1, m2))
            for (l, (Jp1, Jp2)) in enumerate(zip(Jp_list1, Jp_list2))
                m2′ = m2
                for (m1′, v) in nonzero_pairs(Jp1[:, m1])
                    push!(rows, CartesianIndex(l, m1′, m2′))
                    eqs[l, m1′, m2′, m1, m2] += v
                end
                m1′ = m1
                for (m2′, v) in nonzero_pairs(Jp2[:, m2])
                    push!(rows, CartesianIndex(l, m1′, m2′))
                    eqs[l, m1′, m2′, m1, m2] += v
                end
            end
        end
    end
    rows = unique!(sort!(rows))
    reduced_eqs = convert(Array, eqs[rows, cols])
    solutions = try
        _nullspace!(reduced_eqs; atol = TOL_NULLSPACE)
    catch err
        err isa LAPACKException || rethrow(err)
        # try again with more stable algorithm
        @warn "LAPACK SDD failed, retrying with SVD" exception = err
        reduced_eqs = convert(Array, eqs[rows, cols])
        _nullspace!(reduced_eqs; atol = TOL_NULLSPACE, alg = LinearAlgebra.QRIteration())
    end

    N123 = size(solutions, 2)

    @assert N123 == Nsymbol(s1, s2, s3)

    solutions = gaugefix!(solutions)

    CGC = SparseArray{T}(undef, d1, d2, d3, N123)
    for α in 1:N123
        for (i, m1m2) in enumerate(cols)
            # replacing d3 with end fails, because of a subtle sparsearray bug
            CGC[m1m2, d3, α] = solutions[i, α]
        end
    end

    return CGC
end

# the weight `w` shifted by `δ` times the `l`-th simple root, e.g. raised by J⁺ₗ for `δ = 1`
shift_weight(w, l, δ) = Base.setindex(Base.setindex(w, w[l] + δ, l), w[l + 1] - δ, l + 1)

function lower_weight_CGC!(CGC, s1::I, s2::I, s3::I) where {I <: SUNIrrep{N}} where {N}
    N123 = size(CGC, 4)
    T = eltype(CGC)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)
    Jp_cols1 = map(J⁻ -> raising_columns(T, J⁻), Jm_list1)
    Jp_cols2 = map(J⁻ -> raising_columns(T, J⁻), Jm_list2)

    map1 = weightmap(basis(s1))
    map2 = weightmap(basis(s2))
    map3 = weightmap(basis(s3))

    # reverse lexographic order: so all relevant parents should come earlier
    # and should thus have been solved
    w3list = sort(collect(keys(map3)); rev = true)
    wshift = div(nboxes(s1) + nboxes(s2) - nboxes(s3), N)

    # the pairs |m₁, m₂⟩ of the current weight, the parent pairs |m₁′, m₂′, m₃′⟩ with their
    # (known) CGCs, and the terms `(i, parent, pair, J⁻ coefficient)` of the lowering equations
    pairs = CartesianIndex{2}[]
    pairindex = Dict{CartesianIndex{2}, Int}()
    parentpairs = CartesianIndex{3}[]
    terms = Tuple{Int, Int, Int, T}[]
    addpair!(pair) = get!(() -> (push!(pairs, pair); length(pairs)), pairindex, pair)

    # the CGCs of all α are solved for together, as these share the same equations
    for w3 in view(w3list, 2:length(w3list))
        m3list = map3[w3]
        # the states |m₃′⟩ that are lowered to this weight by J⁻ₗ
        parents = [
            (l, m3′) for l in 1:(N - 1)
                for m3′ in get(map3, shift_weight(w3, l, 1), _emptyindexlist)
        ]
        eqs = T[Jm_list3[l][m3, m3′] for (l, m3′) in parents, m3 in m3list]

        # build CGC equations:
        # J⁻₃ |m₃′⟩ = (J⁻₁ ⊗ 𝕀 + 𝕀 ⊗ J⁻₂) |m₁′, m₂′⟩
        empty!(pairs)
        empty!(pairindex)
        empty!(parentpairs)
        empty!(terms)
        for (i, (l, m3′)) in enumerate(parents)
            w3′ = shift_weight(w3, l, 1)
            for (w1′, m1′list) in map1
                w2′ = w3′ .- w1′ .+ wshift
                m2′list = get(map2, w2′, _emptyindexlist)
                isempty(m2′list) && continue
                m1list = get(map1, shift_weight(w1′, l, -1), _emptyindexlist)
                m2list = get(map2, shift_weight(w2′, l, -1), _emptyindexlist)
                for m2′ in m2′list, m1′ in m1′list
                    push!(parentpairs, CartesianIndex(m1′, m2′, m3′))
                    q = length(parentpairs)
                    for m1 in m1list
                        push!(terms, (i, q, addpair!(CartesianIndex(m1, m2′)), Jm_list1[l][m1, m1′]))
                    end
                    for m2 in m2list
                        push!(terms, (i, q, addpair!(CartesianIndex(m1′, m2)), Jm_list2[l][m2, m2′]))
                    end
                end
            end
        end
        parentCGC = T[CGC[parent, α] for parent in parentpairs, α in 1:N123]
        rhs = zeros(T, length(pairs), N123, length(parents))
        for (i, q, p, v) in terms, α in 1:N123
            rhs[p, α, i] += v * parentCGC[q, α]
        end

        # solve equations, together with the raising equations for stability
        sols = lowering_solve(eqs, rhs, raising_gram(T, Jp_cols1, Jp_cols2, pairs))
        for (j, m3) in enumerate(m3list), α in 1:N123, (p, pair) in enumerate(pairs)
            CGC[pair, m3, α] = sols[p, α, j]
        end
    end
    return CGC
end

# Solving the lowering equations `sols * eqsᵀ = rhs` (for every α) on their own is unstable:
# rounding errors along the other irreps in s1 ⊗ s2 grow exponentially from one weight to the
# next. Hence the raising equations J⁺ |m₁, m₂⟩ = J⁺₃ |m₃⟩ (with J⁺ = J⁻ᵀ) are imposed as well,
# which give the same solution in exact arithmetic. The normal equations of both in the
# least-squares sense are `B * sols + sols * A = 2 * rhs * eqs`, with `A = eqsᵀ * eqs` positive
# definite and `B` the Gram matrix of the raising operators on the pairs |m₁, m₂⟩, see
# `raising_gram`. These are solved in the eigenbasis of `A`, with one Cholesky factorization of
# `B + μ * I` for every eigenvalue `μ`.
function lowering_solve(eqs, rhs, B)
    μ, U = eigen!(Symmetric(eqs' * eqs))
    n = size(rhs, 1) * size(rhs, 2)
    sols = reshape(reshape(rhs, n, :) * (2 .* (eqs * U)), size(rhs, 1), size(rhs, 2), :)
    Bμ = similar(B)
    for k in eachindex(μ)
        Bμ .= B
        view(Bμ, diagind(Bμ)) .+= μ[k]
        ldiv!(cholesky!(Symmetric(Bμ)), view(sols, :, :, k))
    end
    return reshape(reshape(sols, n, :) * U', size(sols))
end

# nonzero entries `m′ => J⁺[m′, m]` in every column `m` of the raising operator J⁺ = J⁻ᵀ
function raising_columns(::Type{T}, J⁻) where {T}
    cols = [Pair{Int, T}[] for _ in axes(J⁻, 1)]
    for (I, v) in nonzero_pairs(J⁻)
        push!(cols[I[1]], I[2] => convert(T, v))
    end
    return cols
end

# Gram matrix `B = ∑ₗ J⁺ₗᵀ J⁺ₗ` of the raising operators J⁺ₗ = J⁺₁ ⊗ 𝕀 + 𝕀 ⊗ J⁺₂ restricted to
# the `pairs` |m₁, m₂⟩, which all have the same weight. This includes the J⁺ₗ that map out of the
# weights of s3, whose raising equations simply have a vanishing right hand side.
function raising_gram(::Type{T}, Jp_cols1, Jp_cols2, pairs) where {T}
    # nonzero entries in the rows of all J⁺ₗ, which have distinct images as these differ in weight
    rows = Dict{CartesianIndex{2}, Vector{Pair{Int, T}}}()
    for (p, I) in enumerate(pairs)
        m1, m2 = Tuple(I)
        for (cols1, cols2) in zip(Jp_cols1, Jp_cols2)
            for (m1′, v) in cols1[m1]
                push!(get!(Vector{Pair{Int, T}}, rows, CartesianIndex(m1′, m2)), p => v)
            end
            for (m2′, v) in cols2[m2]
                push!(get!(Vector{Pair{Int, T}}, rows, CartesianIndex(m1, m2′)), p => v)
            end
        end
    end
    B = zeros(T, length(pairs), length(pairs))
    for row in values(rows), (p, v) in row, (p′, v′) in row
        B[p, p′] += v * v′
    end
    return B
end

# Auxiliary tools
function qrpos!(C)
    q, r = qr!(C)
    d = diag(r)
    map!(x -> x == zero(x) ? 1 : sign(x), d, d)
    D = Diagonal(d)
    Q = rmul!(Matrix(q), D)
    R = ldiv!(D, Matrix(r))
    return Q, R
end

function cref!(
        A::AbstractMatrix,
        ɛ = eltype(A) <: Union{Rational, Integer} ? 0 : 10 * length(A) * eps(norm(A, Inf))
    )
    nr, nc = size(A)
    i = j = 1
    @inbounds while i <= nr && j <= nc
        (m, mj) = findabsmax(view(A, i, j:nc))
        mj = mj + j - 1
        if m <= ɛ
            if ɛ > 0
                A[i, j:nc] .= zero(eltype(A))
            end
            i += 1
        else
            @simd for k in i:nr
                A[k, j], A[k, mj] = A[k, mj], A[k, j]
            end
            d = A[i, j]
            @simd for k in i:nr
                A[k, j] /= d
            end
            for k in 1:nc
                if k != j
                    d = A[i, k]
                    @simd for l in i:nr
                        A[l, k] -= d * A[l, j]
                    end
                end
            end
            i += 1
            j += 1
        end
    end
    return A
end

function findabsmax(a)
    isempty(a) && throw(ArgumentError("collection must be non-empty"))
    m = abs(first(a))
    mi = firstindex(a)
    for (k, v) in pairs(a)
        if abs(v) > m
            m = abs(v)
            mi = k
        end
    end
    return m, mi
end

function _nullspace!(
        A::AbstractMatrix; atol::Real = 0.0,
        alg = LinearAlgebra.DivideAndConquer(),
        rtol::Real = (min(size(A)...) * eps(real(float(one(eltype(A)))))) * iszero(atol)
    )
    m, n = size(A)
    (m == 0 || n == 0) && return Matrix{eltype(A)}(I, n, n)
    SVD = svd!(A; full = true, alg)
    tol = max(atol, SVD.S[1] * rtol)
    indstart = sum(s -> s .> tol, SVD.S) + 1
    return copy(SVD.Vt[indstart:end, :]')
end

# remove approximate zeros from sparse array
function purge!(C::SparseArray; atol::Real = TOL_PURGE)
    filter!(((_, v),) -> abs(v) > atol, C.data)
    return C
end
