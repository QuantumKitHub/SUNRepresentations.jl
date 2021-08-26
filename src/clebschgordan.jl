const TOL_NULLSPACE = 1e-13
# tolerance for nullspace
const TOL_GAUGE = 1e-11
# tolerance for gaugefixing should probably be bigger than that with which nullspace was determined

function weightmap(basis)
    N = first(basis).N
    # basis could be a GTPatternIterator{N}, but also a Vector{GTPattern{N}}
    weights = Dict{NTuple{N,Int}, Vector{Int}}()
    for (i, m) in enumerate(basis)
        w = weight(m)
        push!(get!(weights, w, Int[]), i)
    end
    return weights
end

CGCCACHE = Dict{Any, Any}()
CGC(s1::I, s2::I, s3::I) where {I<:SUNIrrep} = CGC(Float64, s1, s2, s3)
function CGC(T::Type{<:Real}, s1::SUNIrrep{N}, s2::SUNIrrep{N}, s3::SUNIrrep{N}) where {N}
    cachetype = Dict{Tuple{SUNIrrep{N}, SUNIrrep{N}, SUNIrrep{N}}, SparseArray{T,4}}
    cache = get!(CGCCACHE, (N, T), cachetype())::cachetype
    return get!(cache, (s1, s2, s3)) do
        _CGC(Float64, s1, s2, s3)
    end
end
function _CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I<:SUNIrrep}
    CGC = highest_weight_CGC(T, s1, s2, s3);
    lower_weight_CGC!(CGC, s1, s2, s3)
    CGC
end

gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))
# gaugefix(C) = C*conj.(first(qrpos!(rref!(permutedims(C)))))

const _emptyindexlist = Vector{Int}()

function highest_weight_CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where I <: SUNIrrep
    d1, d2, d3 = dim(s1), dim(s2), dim(s3)
    N = s1.N

    Jp_list1 = creation(s1)
    Jp_list2 = creation(s2)
    eqs = SparseArray{T}(undef, N-1, d1, d2, d1, d2)

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

    solutions = _nullspace(reduced_eqs; atol = TOL_NULLSPACE)
    N123 = size(solutions, 2)

    @assert N123 == directproduct(s1, s2)[s3]

    solutions = gaugefix!(solutions)

    CGC = SparseArray{T}(undef, d1, d2, d3, N123)
    for α = 1:N123
        for (i, m1m2) in enumerate(cols)
            #replacing d3 with end fails, because of a subtle sparsearray bug
            CGC[m1m2, d3, α] = solutions[i, α]
        end
    end

    return CGC
end

function lower_weight_CGC!(CGC, s1::I, s2::I, s3::I) where I <: SUNIrrep{N} where N
    d1, d2, d3, N123 = size(CGC)
    T = eltype(CGC)
    # we can probably discard the checks; this is an inner method
    # d1, d2, d3 = dim(s1), dim(s2), dim(s3)
    # @assert size(CGC,1) == d1 && size(CGC,2) == d2 && size(CGC,3) == d3
    # N123 = size(CGC,4);

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = weightmap(basis(s1))
    map2 = weightmap(basis(s2))
    map3 = weightmap(basis(s3))
    w3list = sort(collect(keys(map3)); rev = true) # reverse lexographic order
    # if we solve in this order, all relevant parents should come earlier and should thus
    # have been solved

    @assert rem(sum(s1.I) + sum(s2.I) - sum(s3.I), N) == 0 # TODO: remove
    wshift = div(sum(s1.I) + sum(s2.I) - sum(s3.I), N)

    # @threads for α = 1:N123 # TODO: consider multithreaded implementation
    for α = 1:N123
        # TODO: known can be removed, currently checks whether impelmentation is correct
        known = fill(false, d3)
        known[d3] = true

        for w3 in view(w3list, 2:length(w3list))
            m3list = map3[w3]
            jmax = length(m3list)
            imax = sum(1:N-1) do l
                w3′ = Base.setindex(w3, w3[l]+1, l)
                w3′ = Base.setindex(w3′, w3[l+1]-1, l+1)
                return length(get(map3, w3′, _emptyindexlist))
            end
            eqs = Array{T}(undef, (imax, jmax))
            rhs = SparseArray{T}(undef, (imax, d1, d2))
            i = 0
            rows = Vector{CartesianIndex{2}}()
            # build equations
            for (l, (Jm1, Jm2, Jm3)) in enumerate(zip(Jm_list1, Jm_list2, Jm_list3))
                w3′ = Base.setindex(w3, w3[l]+1, l)
                w3′ = Base.setindex(w3′, w3[l+1]-1, l+1)
                for (k, m3′) in enumerate(get(map3, w3′, _emptyindexlist))
                    i += 1
                    for (j, m3) in enumerate(m3list)
                        eqs[i, j] = Jm3[m3, m3′]
                    end
                    @assert known[m3′] # TODO: remove
                    for (w1′, m1′list) in map1
                        w2′ = w3′ .- w1′ .+ wshift
                        m2′list = get(map2, w2′, _emptyindexlist)
                        isempty(m2′list) && continue
                        for m2′ in m2′list, m1′ in m1′list
                            CGCcoeff = CGC[m1′, m2′, m3′, α]
                            # apply Jm1 (annihilator on 1)
                            w1 = Base.setindex(w1′, w1′[l]-1, l)
                            w1 = Base.setindex(w1, w1′[l+1]+1, l+1)
                            for m1 in get(map1, w1, _emptyindexlist)
                                m2 = m2′
                                Jm1coeff = Jm1[m1, m1′]
                                rhs[i, m1, m2] += Jm1coeff*CGCcoeff
                                push!(rows, CartesianIndex(m1, m2))
                            end
                            # apply Jm2
                            w2 = Base.setindex(w2′, w2′[l]-1, l)
                            w2 = Base.setindex(w2, w2′[l+1]+1, l+1)
                            for m2 in get(map2, w2, _emptyindexlist)
                                m1 = m1′
                                Jm2coeff = Jm2[m2, m2′]
                                rhs[i, m1, m2] += Jm2coeff*CGCcoeff
                                push!(rows, CartesianIndex(m1, m2))
                            end
                        end
                    end
                end
            end
            # solve equations
            ieqs = pinv(eqs)
            for (j, m3) in enumerate(m3list)
                @assert !known[m3] # TODO: remove
                @inbounds for Im1m2 in unique!(sort!(rows))
                    for i = 1:imax
                        CGC[Im1m2, m3, α] += ieqs[j, i] * rhs[i, Im1m2]
                    end
                end
                known[m3] = true # TODO: remove
            end
        end
    end
    return CGC
end

# Auxiliary tools
function qrpos!(C)
    q, r = qr(C)
    d = diag(r)
    map!(x-> x == zero(x) ? 1 : sign(x), d, d)
    D = Diagonal(d)
    Q = rmul!(Matrix(q), D)
    R = ldiv!(D, Matrix(r))
    return Q, R
end

function cref!(A::AbstractMatrix,
        ɛ = eltype(A) <: Union{Rational,Integer} ? 0 : 10*length(A)*eps(norm(A, Inf)))
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
            d = A[i,j]
            @simd for k in i:nr
                A[k, j] /= d
            end
            for k in 1:nc
                if k != j
                    d = A[i, k]
                    @simd for l = i:nr
                        A[l, k] -= d*A[l, j]
                    end
                end
            end
            i += 1
            j += 1
        end
    end
    A
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

function _nullspace(A::AbstractMatrix; atol::Real = 0.0, rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(atol))
    m, n = size(A)
    (m == 0 || n == 0) && return Matrix{eltype(A)}(I, n, n)
    SVD = svd(A, full=true, alg = LinearAlgebra.QRIteration())
    tol = max(atol, SVD.S[1]*rtol)
    indstart = sum(s -> s .> tol, SVD.S) + 1
    return copy(SVD.Vt[indstart:end,:]')
end
