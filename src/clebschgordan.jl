const TOL_NULLSPACE = 1e-13
# tolerance for nullspace
const TOL_GAUGE = 1e-11
# tolerance for gaugefixing should probably be bigger than that with which nullspace was determined
const TOL_PURGE = 1e-14
# tolerance for dropping zeros

function weightmap(basis)
    N = first(basis).N
    # basis could be a GTPatternIterator{N}, but also a Vector{GTPattern{N}}
    weights = Dict{NTuple{N,Int},Vector{Int}}()
    for (i, m) in enumerate(basis)
        w = weight(m)
        push!(get!(weights, w, Int[]), i)
    end
    return weights
end

CGC(s1::I, s2::I, s3::I) where {I<:SUNIrrep} = CGC(Float64, s1, s2, s3)
function CGC(::Type{T}, s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where {T,N}
    return _get_CGC(T, (s1, s2, s3))
end

@noinline function _get_CGC(::Type{T}, @nospecialize(key)) where {T}
    d::SparseArray{T,4} = get!(CGC_CACHE, key) do
        result = tryread(T, key...)
        isnothing(result) || return result
        return generate_CGC(T, key...)
    end
    return d
end

function _CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I<:SUNIrrep}
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

gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))

# special case for 1 ⊗ s -> s or s ⊗ 1 -> s
function trivial_CGC(::Type{T}, s::SUNIrrep, isleft=true) where {T<:Real}
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

function highest_weight_CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I<:SUNIrrep}
    d1, d2, d3 = dim(s1), dim(s2), dim(s3)
    N = s1.N

    Jp_list1 = creation(s1)
    Jp_list2 = creation(s2)
    eqs = SparseArray{T}(undef, N - 1, d1, d2, d1, d2)

    cols = Vector{CartesianIndex{2}}()
    rows = Vector{CartesianIndex{3}}()

    map2 = weightmap(basis(s2))
    w3 = weight(highest_weight(s3))
    wshift = div(sum(s1.I) + sum(s2.I) - sum(s3.I), N)

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
        _nullspace!(reduced_eqs; atol=TOL_NULLSPACE)
    catch err
        err isa LAPACKException || rethrow(err)
        # try again with more stable algorithm
        @warn "LAPACK SDD failed, retrying with SVD" exception = err
        reduced_eqs = convert(Array, eqs[rows, cols])
        _nullspace!(reduced_eqs; atol=TOL_NULLSPACE, alg=LinearAlgebra.QRIteration())
    end

    N123 = size(solutions, 2)

    @assert N123 == directproduct(s1, s2)[s3]

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

function lower_weight_CGC!(CGC, s1::I, s2::I, s3::I) where {I<:SUNIrrep{N}} where {N}
    N123 = size(CGC, 4)
    T = eltype(CGC)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = weightmap(basis(s1))
    map2 = weightmap(basis(s2))
    map3 = weightmap(basis(s3))

    # reverse lexographic order: so all relevant parents should come earlier
    # and should thus have been solved
    w3list = sort(collect(keys(map3)); rev=true)

    # precompute some data
    wshift = div(sum(s1.I) + sum(s2.I) - sum(s3.I), N)
    rhs_rows = Int[]
    rhs_cols = CartesianIndex{2}[]
    rhs_vals = T[]

    # @threads for α = 1:N123 # TODO: consider multithreaded implementation
    for α in 1:N123
        for w3 in view(w3list, 2:length(w3list))
            m3list = map3[w3]
            jmax = length(m3list)
            imax = sum(1:(N - 1)) do l
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                return length(get(map3, w3′, _emptyindexlist))
            end
            eqs = Array{T}(undef, (imax, jmax))

            # reset vectors but avoid allocations
            empty!(rhs_rows)
            empty!(rhs_cols)
            empty!(rhs_vals)

            i = 0
            # build CGC equations:
            # J⁻₃ |m₃⟩ = (J⁻₁ ⊗ 𝕀 + 𝕀 ⊗ J⁻₂) |m₁, m₂>
            for (l, (J⁻₁, J⁻₂, J⁻₃)) in enumerate(zip(Jm_list1, Jm_list2, Jm_list3))
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                for m3′ in get(map3, w3′, _emptyindexlist)
                    i += 1
                    for (j, m3) in enumerate(m3list)
                        eqs[i, j] = J⁻₃[m3, m3′]
                    end
                    for (w1′, m1′list) in map1
                        w2′ = w3′ .- w1′ .+ wshift
                        m2′list = get(map2, w2′, _emptyindexlist)
                        isempty(m2′list) && continue
                        for m2′ in m2′list, m1′ in m1′list
                            CGCcoeff = CGC[m1′, m2′, m3′, α]
                            # apply J⁻₁
                            w1 = Base.setindex(w1′, w1′[l] - 1, l)
                            w1 = Base.setindex(w1, w1′[l + 1] + 1, l + 1)
                            for m1 in get(map1, w1, _emptyindexlist)
                                m2 = m2′
                                Jm1coeff = J⁻₁[m1, m1′]
                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm1coeff * CGCcoeff)
                            end
                            # apply J⁻₂
                            w2 = Base.setindex(w2′, w2′[l] - 1, l)
                            w2 = Base.setindex(w2, w2′[l + 1] + 1, l + 1)
                            for m2 in get(map2, w2, _emptyindexlist)
                                m1 = m1′
                                Jm2coeff = J⁻₂[m2, m2′]
                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm2coeff * CGCcoeff)
                            end
                        end
                    end
                end
            end

            # construct dense array for the nonzero columns exclusively
            mask = unique(rhs_cols)
            rhs_cols′ = indexin(rhs_cols, mask)
            rhs = zeros(T, imax, length(mask))
            @inbounds for (row, col, val) in zip(rhs_rows, rhs_cols′, rhs_vals)
                rhs[row, col] += val
            end

            # solve equations
            sols = ldiv!(qr!(eqs), rhs)

            # fill in CGC
            # loop over sols in column major order, CGC is hashmap anyways
            @inbounds for (i, Im1m2) in enumerate(mask)
                for (j, m3) in enumerate(m3list)
                    CGC[Im1m2, m3, α] += sols[j, i]
                end
            end
        end
    end
    return CGC
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

function cref!(A::AbstractMatrix,
               ɛ=eltype(A) <: Union{Rational,Integer} ? 0 :
                 10 * length(A) * eps(norm(A, Inf)))
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

function _nullspace!(A::AbstractMatrix; atol::Real=0.0,
                     alg=LinearAlgebra.DivideAndConquer(),
                     rtol::Real=(min(size(A)...) * eps(real(float(one(eltype(A)))))) *
                                iszero(atol))
    m, n = size(A)
    (m == 0 || n == 0) && return Matrix{eltype(A)}(I, n, n)
    SVD = svd!(A; full=true, alg)
    tol = max(atol, SVD.S[1] * rtol)
    indstart = sum(s -> s .> tol, SVD.S) + 1
    return copy(SVD.Vt[indstart:end, :]')
end

# remove approximate zeros from sparse array
function purge!(C::SparseArray; atol::Real=TOL_PURGE)
    filter!(((_, v),) -> abs(v) > atol, C.data)
    return C
end
