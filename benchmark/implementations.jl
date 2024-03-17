
# first optimization: using dense `eqs` and `rhs` arrays
# also discards checks
function _lower_weight_CGC!(::Val{1}, CGC, s1::SUNIrrep{N}, s2::SUNIrrep{N},
                            s3::SUNIrrep{N}) where {N}
    d1, d2, d3, N123 = size(CGC)
    T = eltype(CGC)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = weightmap(basis(s1))
    map2 = weightmap(basis(s2))
    map3 = weightmap(basis(s3))
    w3list = sort(collect(keys(map3)); rev=true) # reverse lexographic order
    # if we solve in this order, all relevant parents should come earlier and should thus
    # have been solved

    wshift = div(sum(s1.I) + sum(s2.I) - sum(s3.I), N)

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

            rhs_rows = Int[]
            rhs_cols = CartesianIndex{2}[]
            rhs_vals = T[]

            i = 0
            # build equations
            for (l, (Jm1, Jm2, Jm3)) in enumerate(zip(Jm_list1, Jm_list2, Jm_list3))
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                for (k, m3′) in enumerate(get(map3, w3′, _emptyindexlist))
                    i += 1
                    for (j, m3) in enumerate(m3list)
                        eqs[i, j] = Jm3[m3, m3′]
                    end
                    for (w1′, m1′list) in map1
                        w2′ = w3′ .- w1′ .+ wshift
                        m2′list = get(map2, w2′, _emptyindexlist)
                        isempty(m2′list) && continue
                        for m2′ in m2′list, m1′ in m1′list
                            CGCcoeff = CGC[m1′, m2′, m3′, α]
                            # apply Jm1 (annihilator on 1)
                            w1 = Base.setindex(w1′, w1′[l] - 1, l)
                            w1 = Base.setindex(w1, w1′[l + 1] + 1, l + 1)
                            for m1 in get(map1, w1, _emptyindexlist)
                                m2 = m2′
                                Jm1coeff = Jm1[m1, m1′]

                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm1coeff * CGCcoeff)
                            end
                            # apply Jm2
                            w2 = Base.setindex(w2′, w2′[l] - 1, l)
                            w2 = Base.setindex(w2, w2′[l + 1] + 1, l + 1)
                            for m2 in get(map2, w2, _emptyindexlist)
                                m1 = m1′
                                Jm2coeff = Jm2[m2, m2′]

                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm2coeff * CGCcoeff)
                            end
                        end
                    end
                end
            end

            # construct dense array from the nonzero columns exclusively
            mask = unique(rhs_cols)
            rhs_cols′ = indexin(rhs_cols, mask)
            rhs = zeros(T, imax, length(mask))
            @inbounds for (row, col, val) in zip(rhs_rows, rhs_cols′, rhs_vals)
                rhs[row, col] += val
            end

            # solve equations
            sols = ldiv!(qr!(eqs), rhs)

            # fill in CGC
            @inbounds for (i, Im1m2) in enumerate(mask)
                for (j, m3) in enumerate(m3list)
                    CGC[Im1m2, m3, α] += sols[j, i]
                end
            end
        end
    end

    return CGC
end

# optimization round 2: try to re-use arrays
function _lower_weight_CGC!(::Val{2}, CGC, s1::SUNIrrep{N}, s2::SUNIrrep{N},
                            s3::SUNIrrep{N}) where {N}
    d1, d2, d3, N123 = size(CGC)
    T = eltype(CGC)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = weightmap(basis(s1))
    map2 = weightmap(basis(s2))
    map3 = weightmap(basis(s3))
    w3list = sort(collect(keys(map3)); rev=true) # reverse lexographic order
    # if we solve in this order, all relevant parents should come earlier and should thus
    # have been solved

    wshift = div(sum(s1.I) + sum(s2.I) - sum(s3.I), N)

    # @threads for α = 1:N123 # TODO: consider multithreaded implementation
    # pre-allocate some vectors:
    rhs_rows = Int[]
    rhs_cols = CartesianIndex{2}[]
    rhs_vals = T[]

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

            # reset vectors
            empty!(rhs_rows)
            empty!(rhs_cols)
            empty!(rhs_vals)

            i = 0
            # build equations
            for (l, (Jm1, Jm2, Jm3)) in enumerate(zip(Jm_list1, Jm_list2, Jm_list3))
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                for (k, m3′) in enumerate(get(map3, w3′, _emptyindexlist))
                    i += 1
                    for (j, m3) in enumerate(m3list)
                        eqs[i, j] = Jm3[m3, m3′]
                    end
                    for (w1′, m1′list) in map1
                        w2′ = w3′ .- w1′ .+ wshift
                        m2′list = get(map2, w2′, _emptyindexlist)
                        isempty(m2′list) && continue
                        for m2′ in m2′list, m1′ in m1′list
                            CGCcoeff = CGC[m1′, m2′, m3′, α]
                            # apply Jm1 (annihilator on 1)
                            w1 = Base.setindex(w1′, w1′[l] - 1, l)
                            w1 = Base.setindex(w1, w1′[l + 1] + 1, l + 1)
                            for m1 in get(map1, w1, _emptyindexlist)
                                m2 = m2′
                                Jm1coeff = Jm1[m1, m1′]

                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm1coeff * CGCcoeff)
                            end
                            # apply Jm2
                            w2 = Base.setindex(w2′, w2′[l] - 1, l)
                            w2 = Base.setindex(w2, w2′[l + 1] + 1, l + 1)
                            for m2 in get(map2, w2, _emptyindexlist)
                                m1 = m1′
                                Jm2coeff = Jm2[m2, m2′]

                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm2coeff * CGCcoeff)
                            end
                        end
                    end
                end
            end

            # construct dense array from the nonzero columns exclusively
            mask = unique(rhs_cols)
            rhs_cols′ = indexin(rhs_cols, mask)
            rhs = zeros(T, imax, length(mask))
            @inbounds for (row, col, val) in zip(rhs_rows, rhs_cols′, rhs_vals)
                rhs[row, col] += val
            end

            # solve equations
            sols = ldiv!(qr!(eqs), rhs)

            # fill in CGC
            @inbounds for (i, Im1m2) in enumerate(mask)
                for (j, m3) in enumerate(m3list)
                    CGC[Im1m2, m3, α] += sols[j, i]
                end
            end
        end
    end

    return CGC
end

# optimization round 3: try realloc
function _lower_weight_CGC!(::Val{3}, CGC, s1::SUNIrrep{N}, s2::SUNIrrep{N},
                            s3::SUNIrrep{N}) where {N}
    d1, d2, d3, N123 = size(CGC)
    T = eltype(CGC)

    Jm_list1 = annihilation(s1)
    Jm_list2 = annihilation(s2)
    Jm_list3 = annihilation(s3)

    map1 = weightmap(basis(s1))
    map2 = weightmap(basis(s2))
    map3 = weightmap(basis(s3))
    w3list = sort(collect(keys(map3)); rev=true) # reverse lexographic order
    # if we solve in this order, all relevant parents should come earlier and should thus
    # have been solved

    wshift = div(sum(s1.I) + sum(s2.I) - sum(s3.I), N)

    # @threads for α = 1:N123 # TODO: consider multithreaded implementation
    # pre-allocate some vectors:
    rhs_rows = Int[]
    rhs_cols = CartesianIndex{2}[]
    rhs_vals = T[]
    eqs_ptr = Ptr{T}()
    rhs_ptr = Ptr{T}()

    for α in 1:N123
        for w3 in view(w3list, 2:length(w3list))
            m3list = map3[w3]
            jmax = length(m3list)
            imax = sum(1:(N - 1)) do l
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                return length(get(map3, w3′, _emptyindexlist))
            end

            eqs_ptr::Ptr{T} = Base.Libc.realloc(eqs_ptr, sizeof(T) * imax * jmax)
            eqs = unsafe_wrap(Array, eqs_ptr, (imax, jmax))

            # reset vectors
            empty!(rhs_rows)
            empty!(rhs_cols)
            empty!(rhs_vals)

            i = 0
            # build equations
            for (l, (Jm1, Jm2, Jm3)) in enumerate(zip(Jm_list1, Jm_list2, Jm_list3))
                w3′ = Base.setindex(w3, w3[l] + 1, l)
                w3′ = Base.setindex(w3′, w3[l + 1] - 1, l + 1)
                for (k, m3′) in enumerate(get(map3, w3′, _emptyindexlist))
                    i += 1
                    for (j, m3) in enumerate(m3list)
                        eqs[i, j] = Jm3[m3, m3′]
                    end
                    for (w1′, m1′list) in map1
                        w2′ = w3′ .- w1′ .+ wshift
                        m2′list = get(map2, w2′, _emptyindexlist)
                        isempty(m2′list) && continue
                        for m2′ in m2′list, m1′ in m1′list
                            CGCcoeff = CGC[m1′, m2′, m3′, α]
                            # apply Jm1 (annihilator on 1)
                            w1 = Base.setindex(w1′, w1′[l] - 1, l)
                            w1 = Base.setindex(w1, w1′[l + 1] + 1, l + 1)
                            for m1 in get(map1, w1, _emptyindexlist)
                                m2 = m2′
                                Jm1coeff = Jm1[m1, m1′]

                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm1coeff * CGCcoeff)
                            end
                            # apply Jm2
                            w2 = Base.setindex(w2′, w2′[l] - 1, l)
                            w2 = Base.setindex(w2, w2′[l + 1] + 1, l + 1)
                            for m2 in get(map2, w2, _emptyindexlist)
                                m1 = m1′
                                Jm2coeff = Jm2[m2, m2′]

                                push!(rhs_rows, i)
                                push!(rhs_cols, CartesianIndex(m1, m2))
                                push!(rhs_vals, Jm2coeff * CGCcoeff)
                            end
                        end
                    end
                end
            end

            # construct dense array from the nonzero columns exclusively
            mask = unique(rhs_cols)
            rhs_cols′ = indexin(rhs_cols, mask)

            rhs_ptr::Ptr{T} = Base.Libc.realloc(rhs_ptr, sizeof(T) * imax * length(mask))
            rhs = unsafe_wrap(Array, rhs_ptr, (imax, length(mask)))
            fill!(rhs, zero(T))

            @inbounds for (row, col, val) in zip(rhs_rows, rhs_cols′, rhs_vals)
                rhs[row, col] += val
            end

            # solve equations
            sols = ldiv!(qr!(eqs), rhs)

            # fill in CGC
            @inbounds for (i, Im1m2) in enumerate(mask)
                for (j, m3) in enumerate(m3list)
                    CGC[Im1m2, m3, α] += sols[j, i]
                end
            end
        end
    end

    Base.Libc.free(eqs_ptr)
    Base.Libc.free(rhs_ptr)

    return CGC
end
