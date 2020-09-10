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
# you can use search, but this should be faster
function weight2childmap(w3)
    weight2child = Dict{eltype(w3),Vector{Int}}()
    for (i,w) in enumerate(w3)
        if !(w in keys(weight2child))
            weight2child[w] = Int[];
        end
        push!(weight2child[w],i)
    end
    return weight2child;
end

# for every weight it will return all "parent weights", along with how to get there
function weight2parmap(weight2child)
    weight2parent = Dict{eltype(keys(weight2child)),Vector{Tuple{Int,Int}}}();
    for w in keys(weight2child)
        weight2parent[w] = Int[];

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
    prodmap(t) = prodmap(t...);
    prodmap(i,j) = (i-1)*length(p2)+j

    whw = Wz(p3[end]);
    N = p3[end].N

    T = fill(0.0,length(p1)*length(p2),length(p1)*length(p2));
    used_dom = Vector{Tuple{Int64,Int64}}();
    used_codom = Vector{Tuple{Int64,Int64}}(); #this should be a set instead of calling unique on it at the end ...

    #get the allowed basis
    for (i,p1) in enumerate(p1),(j,p2) in enumerate(p2)
        Wz(p1)+Wz(p2) == whw && push!(used_dom,(i,j))
    end

    for (j,k) in used_dom
        for l in 1:N-1
            for (pref,ap1) in creation(p1[j],l)
                x = (findfirst(x->isequal(x,ap1),p1),k);
                push!(used_codom,x)
                T[prodmap(x),prodmap(j,k)] += pref;
            end
            for (pref,ap2) in creation(p2[k],l)
                x = (j,findfirst(x->isequal(x,ap2),p2));
                push!(used_codom,x)
                T[prodmap(x),prodmap(j,k)] += pref;
            end
        end
    end
    T_subslice = T[prodmap.(unique(used_codom)),prodmap.(used_dom)];

    solutions = gauge_fix(LinearAlgebra.nullspace(T_subslice));

    CGC = fill(0.0,length(p3),size(solutions,2),length(p1),length(p2));
    for α in 1:size(solutions,2)
        for (i,(j,k)) in enumerate(used_dom)
            CGC[end,α,j,k] = solutions[i,α]
        end
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
            T = fill(0.0,length(parentbundle),size(CGC,2),length(p1),length(p2));
            for (i,(curpar,l)) in enumerate(parentbundle)
                for (pref,ana) in anihilation(p3[curpar],l)
                    derp = findfirst(x->isequal(p3[x],ana),children);
                    @assert !isnothing(derp);
                    B[i,derp] += pref;
                end

                for α = 1:size(T,2),ip1 = 1:size(T,3),ip2 = 1:size(T,4)
                    cur_CGC = CGC[curpar,α,ip1,ip2];
                    if cur_CGC!=0.0
                        for (pref,ana) in anihilation(p1[ip1],l)
                            derp = findfirst(x->isequal(ana,x),p1)
                            @assert !isnothing(derp);
                            T[i,α,derp,ip2] += pref*cur_CGC;
                        end

                        for (pref,ana) in anihilation(p2[ip2],l)
                            derp = findfirst(x->isequal(ana,x),p2)
                            @assert !isnothing(derp);
                            T[i,α,ip1,derp] += pref*cur_CGC;
                        end
                    end
                end
            end

            @tensor solutions[-1,-2,-3,-4]:=pinv(B)[-1,1]*T[1,-2,-3,-4]
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
