#in the case of multiple fusion, we should gaugefix C
function gauge_fix(C)
    (q,_) = qrpos(rref(permutedims(C)));
    C*conj.(q)
end

function heighest_weight_CGC(p1,p2,p3)
    N = p3[1].N;
    T = SparseArray{Float64}(undef,length(p1),length(p2),length(p1),length(p2),N-1);

    used_dom = Vector{Tuple{Int64,Int64}}();
    used_codom = Vector{Tuple{Int64,Int64,Int64}}();

    for (j,cp1) in enumerate(p1),(k,cp2) in enumerate(p2)
        #this is the selection rule (jz_1 + jz_2 = jz_total)
        Wz(cp1)+Wz(cp2) == Wz(p3[end]) || continue
        push!(used_dom,(j,k))


        for (opp_ind,c1) in enumerate(creation(p1)), (l,v) in TO.nonzeros(c1[:,j])
            push!(used_codom,(l,k,opp_ind))
            T[l,k,j,k,opp_ind] += v;
        end
        for (opp_ind,c2) in enumerate(creation(p2)), (l,v) in TO.nonzeros(c2[:,k])
            push!(used_codom,(j,l,opp_ind))
            T[j,l,j,k,opp_ind] += v;
        end
    end
    used_codom = unique(used_codom);
    dense_T_subslice = zeros(length(used_codom),length(used_dom));
    for (i,(a,b,opp_ind)) in enumerate(used_codom),(j,(c,d)) in enumerate(used_dom)
        dense_T_subslice[i,j] = T[a,b,c,d,opp_ind];
    end

    solutions = gauge_fix(LinearAlgebra.nullspace(dense_T_subslice));

    CGC = SparseArray{Float64}(undef,length(p3),size(solutions,2),length(p1),length(p2));

    for α in 1:size(solutions,2),(i,(j,k)) in enumerate(used_dom)
        CGC[end,α,j,k] = solutions[i,α]
    end

    return CGC
end

function lower_weight_CGC!(CGC,p1,p2,p3)
    known = fill(false,length(p3));
    child2parmap = Dict{Int64,Vector{Tuple{Int64,Int64,Float64}}}();
    N = p3[1].N;

    function graduate!(new_parent)
        delete!(child2parmap,new_parent);
        known[new_parent] = true;

        for (j1,ana) in enumerate(anihilation(p3)),(new_child,val) in TO.nonzeros(ana[:,new_parent])

            cur = Vector{Tuple{Int64,Int64,Float64}}();
            for (j2,crea) in enumerate(creation(p3)),(other_parent,tval) in TO.nonzeros(crea[:,new_child])
                push!(cur,(other_parent,j2,conj(tval)));
            end
            child2parmap[new_child] = cur;
        end
    end

    graduate!(length(p3));
    while !isempty(child2parmap)
        curent_class = Int64[];
        sparse2dense = SparseArray{Int64}(undef,length(p3),N-1);
        dense_len = 0;

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
        B = fill(0.0,dense_len,length(curent_class));
        for (child_index,child) in enumerate(curent_class),
            (parent,j,val) in child2parmap[child]

            B[sparse2dense[parent,j],child_index] = val;
        end

        #build T
        T = SparseArray{Float64}(undef,dense_len,size(CGC,2),length(p1),length(p2));
        for (index,val) in TO.nonzeros(sparse2dense)
            (parent,j) = Tuple(index);

            for α = 1:size(T,2),ip1 = 1:size(T,3),ip2 = 1:size(T,4)
                cur_CGC = CGC[parent,α,ip1,ip2];
                cur_CGC == zero(cur_CGC) && continue;

                for (derp,pref) in TO.nonzeros(anihilation(p1)[j][:,ip1])
                    T[val,α,derp,ip2] += pref*cur_CGC;
                end

                for (derp,pref) in TO.nonzeros(anihilation(p2)[j][:,ip2])
                    pref == zero(pref) && continue;
                    T[val,α,ip1,derp] += pref*cur_CGC;
                end
            end
        end

        # pinv(B) * T = CGC (we could also use KrylovKit)
        @tensor solutions[-1,-2,-3,-4] := TensorOperations.SparseArray(pinv(B))[-1,1]*T[1,-2,-3,-4]
        for (i,c) in enumerate(curent_class)
            CGC[c,:,:,:] = solutions[i,:,:,:]
            graduate!(c);
        end
    end
end
#actually calculating the CGC's
function CGC(s1::SUNIrrep{N},s2::SUNIrrep{N},s3::SUNIrrep{N},p1 = GTpatterns(s1),p2 = GTpatterns(s2),p3 = GTpatterns(s3)) where N
    CGC = heighest_weight_CGC(p1,p2,p3);
    lower_weight_CGC!(CGC,p1,p2,p3)
    CGC
end
