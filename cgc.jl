function qrpos(C)
    (q,r) = LinearAlgebra.qr(C);
    D = LinearAlgebra.diagm(sign.(LinearAlgebra.diag(q)));
    (q*D,D*r)
end
function lqpos(C)
    (l,q) = LinearAlgebra.lq(C);
    D = LinearAlgebra.diagm(sign.(LinearAlgebra.diag(q)));
    (l*D,D*q)
end
#in the case of multiple fusion, we should gaugefix C
function gauge_fix(C)
    (q,_) = qrpos(rref(permutedims(C)));
    C*conj.(q)
end

# maps a given weight to the (index of) the patterns with that weight
function weight2childmap(w3)
    weight2child = Dict((w=>Int[] for w in w3))
    for (i,w) in enumerate(w3)
        push!(weight2child[w],i)
    end
    return weight2child;
end

# for every weight it will return all "parent weights", along with how to get there
function weight2parmap(weight2child)
    weight2parent = Dict((w=>Vector{Tuple{Int,Int}}() for w in keys(weight2child)));
    for w in keys(weight2child)
        for l in 1:length(w)-1
            d = fill(0,length(w));
            d[l] = 1;
            d[l+1] = -1;

            if w+d in keys(weight2child)
                append!(weight2parent[w],zip(weight2child[w+d],fill(l,length(weight2child[w+d]))))
            end
        end

    end
    return weight2parent
end

function heighest_weight_CGC(p1,p2,p3)
    T = SparseArray{Float64}(undef,length(p1),length(p2),length(p1),length(p2));

    used_dom = Vector{Tuple{Int64,Int64}}();
    used_codom = Vector{Tuple{Int64,Int64}}();

    for (j,cp1) in enumerate(p1),(k,cp2) in enumerate(p2)
        #this is the selection rule (jz_1 + jz_2 = jz_total)
        Wz(cp1)+Wz(cp2) == Wz(p3[end]) || continue
        push!(used_dom,(j,k))

        for c1 in creation(p1), (l,v) in enumerate(c1[:,j])
            push!(used_codom,(l,k))
            T[l,k,j,k] += v;
        end
        for c2 in creation(p2), (l,v) in enumerate(c2[:,k])
            push!(used_codom,(j,l))
            T[j,l,j,k] += v;
        end
    end
    used_codom = unique(used_codom);
    dense_T_subslice = zeros(length(used_codom),length(used_dom));
    for (i,(a,b)) in enumerate(used_codom),(j,(c,d)) in enumerate(used_dom)
        dense_T_subslice[i,j] = T[a,b,c,d];
    end

    solutions = gauge_fix(LinearAlgebra.nullspace(dense_T_subslice));

    CGC = SparseArray{Float64}(undef,length(p3),size(solutions,2),length(p1),length(p2));

    for α in 1:size(solutions,2),(i,(j,k)) in enumerate(used_dom)
        CGC[end,α,j,k] = solutions[i,α]
    end

    return CGC
end

function lower_weight_CGC!(CGC,p1,p2,p3)
    wp3 = W.(p3);

    weight2child = weight2childmap(wp3);
    weight2parent = weight2parmap(weight2child);

    #we assume that we know the heighest weight cGC
    known = fill(false,length(p3));
    known[end] = true;

    @assert isempty(weight2parent[wp3[end]]) # there are no parents of the largest weight irrep
    delete!(weight2parent,wp3[end]);

    infloop = false
    while !infloop
        infloop = true;

        for (k,parentbundle) in weight2parent

            #we don't know all parents
            !reduce(&,map(x->known[x[1]],parentbundle)) && continue

            children = weight2child[k];
            B = fill(0.0,length(parentbundle),length(children));
            T = TensorOperations.SparseArray{Float64}(undef,(length(parentbundle),size(CGC,2),length(p1),length(p2)));
            for (i,(curpar,l)) in enumerate(parentbundle)


                for (ana,pref) in enumerate(anihilation(p3)[l][:,curpar])
                    pref == zero(pref) && continue;
                    B[i,findfirst(x->isequal(x,ana),children)] += pref;
                end

                for α = 1:size(T,2),ip1 = 1:size(T,3),ip2 = 1:size(T,4)
                    cur_CGC = CGC[curpar,α,ip1,ip2];
                    cur_CGC == zero(cur_CGC) && continue;

                    for (derp,pref) in enumerate(anihilation(p1)[l][:,ip1])
                        T[i,α,derp,ip2] += pref*cur_CGC;
                    end

                    for (derp,pref) in enumerate(anihilation(p2)[l][:,ip2])
                        T[i,α,ip1,derp] += pref*cur_CGC;
                    end
                end
            end
            #as alternative to pinv we can use krylovkit?
            @tensor solutions[-1,-2,-3,-4] := TensorOperations.SparseArray(pinv(B))[-1,1]*T[1,-2,-3,-4]
            for (i,c) in enumerate(children)
                CGC[c,:,:,:] = solutions[i,:,:,:]
            end

            infloop = false;
            known[children].=true;
            delete!(weight2parent,k);
            break;
        end
    end
end
#actually calculating the CGC's
function CGC(s1::SUNIrrep{N},s2::SUNIrrep{N},s3::SUNIrrep{N},p1 = GTpatterns(s1),p2 = GTpatterns(s2),p3 = GTpatterns(s3)) where N
    CGC = heighest_weight_CGC(p1,p2,p3);
    lower_weight_CGC!(CGC,p1,p2,p3)
    CGC
end
