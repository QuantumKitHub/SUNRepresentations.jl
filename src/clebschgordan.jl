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
    if isunit(s1)
        @assert s2 == s3
        CGC = trivial_CGC(T, s2, true)
    elseif isunit(s2)
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
    # the nonzero entries in the columns of J⁻ₗ and of J⁺ₗ = J⁻ₗᵀ
    Jm_cols1 = map(J⁻ -> nonzero_lists(T, J⁻; dims = 2), Jm_list1)
    Jm_cols2 = map(J⁻ -> nonzero_lists(T, J⁻; dims = 2), Jm_list2)
    Jm_cols3 = map(J⁻ -> nonzero_lists(T, J⁻; dims = 2), annihilation(s3))
    Jp_cols1 = map(J⁻ -> nonzero_lists(T, J⁻; dims = 1), Jm_list1)
    Jp_cols2 = map(J⁻ -> nonzero_lists(T, J⁻; dims = 1), Jm_list2)

    map1 = weightmap(basis(s1))
    map2 = weightmap(basis(s2))
    map3 = weightmap(basis(s3))

    # reverse lexographic order: so all relevant parents should come earlier
    # and should thus have been solved
    w3list = sort(collect(keys(map3)); rev = true)
    wshift = div(nboxes(s1) + nboxes(s2) - nboxes(s3), N)

    # the CGCs of every solved weight, as its pairs |m₁, m₂⟩, their indices, and an array
    # `sols[p, α, j]` for the `j`-th state of that weight, starting from the highest weight
    solved = Dict{NTuple{N, Int}, Tuple{Vector{CartesianIndex{2}}, Dict{CartesianIndex{2}, Int}, Array{T, 3}}}()
    pairs, pairindex = CartesianIndex{2}[], Dict{CartesianIndex{2}, Int}()
    for (ind, _) in nonzero_pairs(CGC)
        add_pair!(pairs, pairindex, CartesianIndex(ind[1], ind[2]))
    end
    sols = zeros(T, length(pairs), N123, 1)
    for (ind, v) in nonzero_pairs(CGC)
        sols[pairindex[CartesianIndex(ind[1], ind[2])], ind[4], 1] = v
    end
    solved[first(w3list)] = (pairs, pairindex, sols)

    # the parent states `(l, m₃′)` of the current weight with the CGCs `sols′[p′, α, j′]` of their
    # pairs as `(sols′, j′)`, and the terms `(i, p′, p, J⁻ coefficient)` of the lowering equations
    parents = Tuple{Int, Int}[]
    parentsols = Tuple{Array{T, 3}, Int}[]
    terms = Tuple{Int, Int, Int, T}[]
    # buffers that are reused for all weights
    eqsbuffer, rhsbuffer, Bcbuffer = T[], T[], T[]

    # the CGCs of all α are solved for together, as these share the same equations
    for w3 in view(w3list, 2:length(w3list))
        m3list = map3[w3] # sorted
        # the states |m₃′⟩ that are lowered to this weight by J⁻ₗ
        empty!(parents)
        for l in 1:(N - 1), m3′ in get(map3, shift_weight(w3, l, 1), _emptyindexlist)
            push!(parents, (l, m3′))
        end
        eqs = fill!(reuse_buffer!(eqsbuffer, length(parents), length(m3list)), zero(T))
        for (i, (l, m3′)) in enumerate(parents), (m3, v) in Jm_cols3[l][m3′]
            eqs[i, searchsortedfirst(m3list, m3)] = v
        end

        # build CGC equations:
        # J⁻₃ |m₃′⟩ = (J⁻₁ ⊗ 𝕀 + 𝕀 ⊗ J⁻₂) |m₁′, m₂′⟩
        # pairs that only appear with a vanishing coefficient have vanishing CGCs, and are omitted,
        # whereas parent pairs that were omitted contribute nothing
        pairs, pairindex = CartesianIndex{2}[], Dict{CartesianIndex{2}, Int}()
        empty!(parentsols)
        empty!(terms)
        for (i, (l, m3′)) in enumerate(parents)
            w3′ = shift_weight(w3, l, 1)
            _, pairindex′, sols′ = solved[w3′]
            push!(parentsols, (sols′, searchsortedfirst(map3[w3′], m3′)))
            for (w1′, m1′list) in map1
                m2′list = get(map2, w3′ .- w1′ .+ wshift, _emptyindexlist)
                for m2′ in m2′list, m1′ in m1′list
                    p′ = get(pairindex′, CartesianIndex(m1′, m2′), 0)
                    for (m1, v) in Jm_cols1[l][m1′]
                        p = add_pair!(pairs, pairindex, CartesianIndex(m1, m2′))
                        p′ == 0 || push!(terms, (i, p′, p, v))
                    end
                    for (m2, v) in Jm_cols2[l][m2′]
                        p = add_pair!(pairs, pairindex, CartesianIndex(m1′, m2))
                        p′ == 0 || push!(terms, (i, p′, p, v))
                    end
                end
            end
        end
        np = length(pairs)
        rhs = fill!(reuse_buffer!(rhsbuffer, np, N123, length(parents)), zero(T))
        for (i, p′, p, v) in terms
            sols′, j′ = parentsols[i]
            for α in 1:N123
                rhs[p, α, i] += v * sols′[p′, α, j′]
            end
        end

        # solve equations, together with the raising equations for stability
        B = raising_gram(T, Jp_cols1, Jp_cols2, pairs)
        Bc = T === Float64 && np >= SPARSE_CHOLESKY_SIZE ? nothing : reuse_buffer!(Bcbuffer, np, np)
        sols = Array{T, 3}(undef, np, N123, length(m3list))
        lowering_solve!(sols, Bc, eqs, rhs, B)
        solved[w3] = (pairs, pairindex, sols)
    end

    # entries that `purge!` would remove anyway are not written
    sizehint!(CGC.data, sum(length ∘ last, values(solved)))
    for (w3, (pairs, _, sols)) in solved
        for (j, m3) in enumerate(map3[w3]), α in 1:N123, (p, pair) in enumerate(pairs)
            v = sols[p, α, j]
            abs(v) > TOL_PURGE && (CGC[pair, m3, α] = v)
        end
    end
    return CGC
end

# Solve the lowering equations `sols * eqsᵀ = rhs` together with the raising equations, which
# have the same solution in exact arithmetic but prevent rounding errors from growing from one
# weight to the next. As both are consistent, any weighting gives the same least-squares solution:
# weighting the lowering equations with `c * eqs * A⁻² * eqsᵀ`, where `A = eqsᵀ * eqs`, yields
# `(B + c * I) * sols = rhs * eqs * (I + c * A⁻¹)`, with `B` from `raising_gram`. Choosing `c` as
# the mean eigenvalue of `A` keeps both sets of equations equally important. `Bc` is workspace.
function lowering_solve!(sols, Bc, eqs, rhs, B)
    A = Symmetric(eqs' * eqs)
    c = tr(A) / size(A, 1)
    M = rmul!(inv(cholesky(A)), c) + I
    mul!(reshape(sols, :, size(M, 1)), reshape(rhs, :, size(rhs, 3)), eqs * M)
    X = reshape(sols, size(sols, 1), :)
    if isnothing(Bc)
        # a sparse Cholesky (CHOLMOD, hence only for `Float64`) for large blocks, as the fill-in is small
        X .= cholesky(Symmetric(B); shift = c) \ X
    else
        copyto!(Bc, B)
        view(Bc, diagind(Bc)) .+= c
        ldiv!(cholesky!(Symmetric(Bc)), X)
    end
    return sols
end
# the number of pairs from which on a sparse Cholesky is used
const SPARSE_CHOLESKY_SIZE = 100

# the index of `pair` in `pairs`, to which it is added if necessary
add_pair!(pairs, pairindex, pair) = get!(() -> (push!(pairs, pair); length(pairs)), pairindex, pair)

# an array of size `dims` that uses (and if necessary enlarges) the memory of `buffer`
function reuse_buffer!(buffer::Vector, dims::Int...)
    n = prod(dims)
    length(buffer) < n && resize!(buffer, n)
    return reshape(view(buffer, 1:n), dims)
end

# nonzero entries of the sparse matrix `A` in every column `j`, as `i => A[i, j]`, or in every
# row `i` for `dims = 1`, as `j => A[i, j]`
function nonzero_lists(::Type{T}, A; dims::Int) where {T}
    lists = [Pair{Int, T}[] for _ in axes(A, dims)]
    for (I, v) in nonzero_pairs(A)
        push!(lists[I[dims]], I[3 - dims] => convert(T, v))
    end
    return lists
end

# Gram matrix `B = ∑ₗ J⁺ₗᵀ J⁺ₗ` of the raising operators J⁺ₗ = J⁺₁ ⊗ 𝕀 + 𝕀 ⊗ J⁺₂ restricted to
# the `pairs` |m₁, m₂⟩, which all have the same weight, as a sparse matrix. This includes the J⁺ₗ
# that map out of the weights of s3, whose raising equations simply have a vanishing right hand side.
function raising_gram(::Type{T}, Jp_cols1, Jp_cols2, pairs) where {T}
    # `R[p, q] = ⟨q| J⁺ₗ |p⟩`, with the images `q` of all J⁺ₗ, which are distinct as these differ in weight
    images = Dict{CartesianIndex{2}, Int}()
    ps, qs, vs = Int[], Int[], T[]
    addimage!(image) = get!(images, image, length(images) + 1)
    for (p, pair) in enumerate(pairs)
        m1, m2 = Tuple(pair)
        for (cols1, cols2) in zip(Jp_cols1, Jp_cols2)
            for (m1′, v) in cols1[m1]
                push!(ps, p)
                push!(qs, addimage!(CartesianIndex(m1′, m2)))
                push!(vs, v)
            end
            for (m2′, v) in cols2[m2]
                push!(ps, p)
                push!(qs, addimage!(CartesianIndex(m1, m2′)))
                push!(vs, v)
            end
        end
    end
    R = sparse(ps, qs, vs, length(pairs), length(images))
    return R * R'
end

# Auxiliary tools
function qrpos!(C)
    q, r = qr!(C)
    d = diag(r)
    map!(x -> iszero(x) ? one(x) : sign(x), d, d)
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
    # rows with entries below `ɛ` are not used as pivots, but are kept: as only column operations
    # are applied to all rows, the columns of `A` keep spanning the same space
    @inbounds while i <= nr && j <= nc
        (m, mj) = findabsmax(view(A, i, j:nc))
        mj = mj + j - 1
        if m <= ɛ
            i += 1
        else
            @simd for k in 1:nr
                A[k, j], A[k, mj] = A[k, mj], A[k, j]
            end
            d = A[i, j]
            @simd for k in 1:nr
                A[k, j] /= d
            end
            for k in 1:nc
                if k != j
                    d = A[i, k]
                    @simd for l in 1:nr
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
