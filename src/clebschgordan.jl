function Z2weightmap(basis)
    N = first(basis).N
    # basis could be a GTPatternIterator{N}, but also a Vector{GTPattern{N}}
    weights = Dict{NTuple{N-1,Int}, Vector{Int}}()
    for (i, m) in enumerate(basis)
        w = Z2weight(m)
        push!(get!(weights, w, Int[]), i)
    end
    return weights
end
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
CGC(s1::I, s2::I, s3::I) where {I<:Irrep} = CGC(Float64, s1, s2, s3)
function CGC(T::Type{<:Real}, s1::Irrep{N}, s2::Irrep{N}, s3::Irrep{N}) where {N}
    cachetype = Dict{Tuple{Irrep{N}, Irrep{N}, Irrep{N}}, SparseArray{T,4}}
    cache = get!(CGCCACHE, (N, T), cachetype())::cachetype
    return get!(cache, (s1, s2, s3)) do
        _CGC(Float64, s1, s2, s3)
    end
end
function _CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I<:Irrep}
    CGC = highest_weight_CGC(T, s1, s2, s3);
    lower_weight_CGC!(CGC, s1, s2, s3)
    CGC
end

#gaugefix(C) = first(qrpos!(transpose(rref!(transpose(C)))))
gaugefix(C) = C*conj.(first(qrpos!(rref!(permutedims(C)))))

const _emptyindexlist = Vector{Int}()

function highest_weight_CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I<:Irrep}
    d1, d2, d3 = dimension(s1), dimension(s2), dimension(s3)
    N = s1.N

    Jp_list1 = creation(s1)
    Jp_list2 = creation(s2)
    eqs = SparseArray{T}(undef, N-1, d1, d2, d1, d2)

    cols = Vector{CartesianIndex{2}}()
    rows = Vector{CartesianIndex{3}}()

    basis2 = collect(basis(s2))
    map2 = Z2weightmap(basis2)
    m3 = highest_weight(s3)
    λ3 = Z2weight(m3)
    for (j1, m1) in enumerate(basis(s1))
        λ1 = Z2weight(m1)
        λ2 = λ3 .- λ1
        for j2 in get(map2, λ2, _emptyindexlist)
            m2 = basis2[j2]
            push!(cols, CartesianIndex(j1, j2))
            for (l, (Jp1, Jp2)) in enumerate(zip(Jp_list1, Jp_list2))
                i2 = j2
                for (i1, v) in nonzero_pairs(Jp1[:, j1])
                    push!(rows, CartesianIndex(l, i1, i2))
                    eqs[l, i1, i2, j1, j2] += v
                end
                i1 = j1
                for (i2, v) in nonzero_pairs(Jp2[:, j2])
                    push!(rows, CartesianIndex(l, i1, i2))
                    eqs[l, i1, i2, j1, j2] += v
                end
            end
        end
    end
    rows = unique!(sort!(rows))
    reduced_eqs = convert(Array, view(eqs, rows, cols))

    solutions = nullspace(reduced_eqs)
    N123 = size(solutions, 2)

    @assert N123 == directproduct(s1, s2)[s3]

    solutions = gaugefix(solutions)

    CGC = SparseArray{T}(undef, d1, d2, d3, N123)
    for α = 1:N123
        for (i, j1j2) in enumerate(cols)
            #replacing d3 with end fails, because of a subtle sparsearray bug
            CGC[j1j2, d3, α] = solutions[i, α]
        end
    end

    return CGC
end


function lower_weight_CGC!(CGC, s1::I, s2::I, s3::I) where I<: Irrep{N} where N
    d1, d2, d3, N123 = size(CGC)
    T = eltype(CGC)
    # we can probably discard the checks; this is an inner method
    # d1, d2, d3 = dimension(s1), dimension(s2), dimension(s3)
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
                    for w1′ in keys(map1)
                        w2′ = w3′ .- w1′ .+ wshift
                        haskey(map2, w2′) || continue
                    # TODO: switch depending on length(map1) > or < length(map2)
                    # for w2′ in keys(map2)
                    #     w1′ = w3′ .- w2′ .+ wshift
                    #     haskey(map1, w1′) || continue
                        for m1′ in map1[w1′], m2′ in map2[w2′]
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

function rref!(A::AbstractMatrix, ɛ = eltype(A) <: Union{Rational,Integer} ? 0 : eps(norm(A, Inf)))
    nr, nc = size(A)
    i = j = 1
    @inbounds while i <= nr && j <= nc
        (m, mi) = findabsmax(view(A, i:nr, j))
        mi = mi + i - 1
        if m <= ɛ
            if ɛ > 0
                A[i:nr, j] .= zero(eltype(A))
            end
            j += 1
        else
            for k in j:nc
                A[i, k], A[mi, k] = A[mi, k], A[i, k]
            end
            d = A[i,j]
            for k in j:nc
                A[i, k] /= d
            end
            for k in 1:nr
                if k != i
                    d = A[k,j]
                    for l = j:nc
                        A[k,l] -= d*A[i,l]
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
