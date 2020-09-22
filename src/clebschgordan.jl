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

CGC(s1::I, s2::I, s3::I) where {I<:Irrep} = CGC(Float64, s1, s2, s3)

function CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I<:Irrep}
    CGC = highest_weight_CGC(T, s1, s2, s3);
    lower_weight_CGC!(CGC, s1, s2, s3)
    CGC
end

#gaugefix(C) = first(qrpos!(transpose(rref!(transpose(C)))))
gaugefix(C) = C*conj.(first(qrpos!(rref!(permutedims(C)))))

function highest_weight_CGC(T::Type{<:Real}, s1::I, s2::I, s3::I) where {I<:Irrep}
    d1, d2, d3 = dimension(s1), dimension(s2), dimension(s3)
    N = s1.N

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
        for j2 in get(map2, λ2, Vector{Int}())
            m2 = basis2[j2]
            push!(cols, CartesianIndex(j1, j2))
            for (l, (Jp1, Jp2)) in enumerate(zip(creation(s1), creation(s2)))
                i2 = j2
                for (i1, v) in nonzeros(Jp1[:, j1])
                    push!(rows, CartesianIndex(l, i1, i2))
                    eqs[l, i1, i2, j1, j2] += v
                end
                i1 = j1
                for (i2, v) in nonzeros(Jp2[:, j2])
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
            CGC[j1j2, end, α] = solutions[i, α]
        end
    end

    return CGC
end


function lower_weight_CGC!(CGC,s1::I,s2::I,s3::I) where I<: Irrep{N} where N
    d1, d2, d3 = dimension(s1), dimension(s2), dimension(s3)
    @assert size(CGC,1) == d1 && size(CGC,2) == d2 && size(CGC,3) == d3
    N123 = size(CGC,4);

    #indicates whether the given node is known
    known = fill(false,d3);

    #given a child, maps to (par,J,fact) where <par| S_j^+ |child> = fact
    child2parmap = Dict{Int64,Vector{Tuple{Int64,Int64,Float64}}}();

    # mark the given element as fully solved
    function graduate!(new_parent)
        delete!(child2parmap,new_parent);
        known[new_parent] = true;

        for (j1,ana) in enumerate(annihilation(s3)),(new_child,val) in nonzeros(ana[:,new_parent])

            cur = Vector{Tuple{Int64,Int64,Float64}}();
            for (j2,crea) in enumerate(creation(s3)),(other_parent,tval) in nonzeros(crea[:,new_child])
                push!(cur,(other_parent,j2,conj(tval)));
            end
            child2parmap[new_child] = cur;
        end
    end

    graduate!(d3);
    while !isempty(child2parmap)

        curent_class = Int64[]; #these are the kids we will solve for
        sparse2dense = SparseArray{Int64}(undef,d3,N-1); #maps (parent,J) to it's dense index
        dense_len = 0; #maximum(sparse2dense[:])

        #for every child - parents combo:
        #if all parents are known - add child 2 class
        for (k,v) in child2parmap
            if reduce(&,map(x->known[x[1]],v)) # all parents are known
                push!(curent_class,k);

                #we need to map parent - j combos to an index (this effectively indexes the system of equations)
                for (p,j,_) in v
                    if sparse2dense[p,j] == zero(sparse2dense[p,j])
                        dense_len += 1;
                        sparse2dense[p,j] = dense_len
                    end
                end
            end
        end

        #we cannot (even though we should be able to) solve the system of equations using the current approach
        #instead of inflooping, we throw an error (tough this should never ever happen)
        isempty(curent_class) && throw(ArgumentError("disconnected"));

        #build B
        B = fill(zero(eltype(CGC)),dense_len,length(curent_class));
        for (child_index,child) in enumerate(curent_class),
            (parent,j,val) in child2parmap[child]

            B[sparse2dense[parent,j],child_index] = val;
        end

        #build T
        T = SparseArray{eltype(CGC)}(undef,d1,d2,dense_len,N123);
        for (index,val) in nonzeros(sparse2dense)
            (parent,j) = Tuple(index);

            for ip1 = 1:d1,ip2 = 1:d2,α = 1:N123
                cur_CGC = CGC[ip1,ip2,parent,α];
                cur_CGC == zero(cur_CGC) && continue;

                for (derp,pref) in nonzeros(annihilation(s1)[j][:,ip1])
                    T[derp,ip2,val,α] += pref*cur_CGC;
                end

                for (derp,pref) in nonzeros(annihilation(s2)[j][:,ip2])
                    T[ip1,derp,val,α] += pref*cur_CGC;
                end
            end
        end
        # pinv(B) * T = CGC (we could also use KrylovKit)
        @tensor solutions[-1,-2,-3,-4] := SparseArray(pinv(B))[-3,1]*T[-1,-2,1,-4]

        for (i,c) in enumerate(curent_class)
            CGC[:,:,c,:] = solutions[:,:,i,:]
            graduate!(c);
        end
    end
end
# Auxiliary tools
function qrpos!(C)
    q, r = qr(C)
    d = diag(r)
    map!(sign, d, d)
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
