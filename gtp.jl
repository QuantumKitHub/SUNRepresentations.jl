#=
GTP = gelfand-Tsetlin pattern

things to change
    - data container in GTPattern
    - some names
    - if I implement the indexing from the paper, the CGC routine will be so much nicer (no more findindex)

However, runtime should be in the matrix inversion / nullspace subroutines anyway
=#

struct GTPattern
    data::Matrix{Int64}
end
Base.getproperty(d::GTPattern,s::Symbol) = s == :N ? size(d.data,1) : getfield(d,s);

Base.getindex(d::GTPattern,i,j) = d.data[i,j]
Base.checkbounds(d::GTPattern,i,j) = 0<i<=j<=d.N
function Base.show(io::IO,d::GTPattern)
    println(io,"GTpattern{$(size(d.data,1))}:")
    for r = 1:d.N
        str = reduce((a,b) -> string(a," ",b),string.(d.data[1:end-r+1,end-r+1]))

        padding = repeat(" ",r-1);

        print(io,padding);
        println(io,str);

    end
end
function Base.isless(a::GTPattern,b::GTPattern)
    @assert a.N == b.N #what are you comparing?
    for l in a.N:-1:1,k in 1:l
        a[k,l] > b[k,l] && return false;
    end
    return true
end

Base.hash(d::GTPattern,h::UInt) = hash(d.data[:],h)
Base.isequal(a::GTPattern,b::GTPattern) = a.data==b.data
rowsum(d::GTPattern,i) = sum(d.data[1:i,i])

#the Z weight
function Wz(d::GTPattern)
    w = [0;[rowsum(d,i) for i in 1:d.N]];
    lambdas = w[2:end-1]-(w[3:end]+w[1:end-2]).*0.5

end

#the pattern weight
function W(d::GTPattern)
    w = [rowsum(d,i) for i in 1:d.N];
    w[2:end]-=w[1:end-1]
    w
end

#todo - invent a better name
function GT_pattern_down(irr::SUNIrrep{N}) where N
    @assert N>1

    toret = collect(Iterators.product(ntuple(i->irr.s[i+1]:irr.s[i],N-1)...))
    SUNIrrep.(toret[:])
end

function GTpatterns(irr::SUNIrrep{N}) where N
    N == 1 && return [GTPattern(fill(irr.s[1],1,1))]

    patterns::Vector{GTPattern} = reduce(vcat,GTpatterns.(GT_pattern_down(irr)));

    patterns = map(patterns) do pat
        #we have the paterns from a descendent of irr, so now we insert irr itself
        GTPattern(hcat(vcat(pat.data,fill(0,1,N-1)),collect(irr.s)))
    end

    sort(patterns)
end

function highest_weight_GT(irr::SUNIrrep{N}) where N
    data = Matrix{Int64}(undef,N,N);
    for k = 1:N
        data[k,k:end].=irr.s[k];
        data[k,1:k-1].=0
    end
    GTPattern(data)
end

#this has to be a base function somewhere
function countpartition(set)
    toret = Dict{eltype(set),Int}();
    for s in set
        toret[s] = get(toret,s,0)+1
    end
    return toret
end

function creation(p1::GTPattern,l)
    @assert l >=1 && l <=p1.N-1
    result = Tuple{ComplexF64,typeof(p1)}[];
    for k = 1:l
        a = prod((p1[kp,l+1]-p1[k,l]+k-kp for kp in 1:l+1));
        b = l>1 ? prod((p1[kp,l-1]-p1[k,l]+k-kp-1 for kp in 1:l-1)) : 1;
        c = prod((kp==k ? 1 : (p1[kp,l] - p1[k,l] + k - kp)*(p1[kp,l]-p1[k,l]+k-kp-1) for kp in 1:l))

        pref = sqrt((-1.0+0im)*a*b/c)
        if pref != 0 && c != 0
            td = copy(p1.data);
            td[k,l]+=1;
            push!(result,(pref,GTPattern(td)));
        end
    end

    return result
end

function anihilation(p1::GTPattern,l)
    @assert l >=1 && l <=p1.N-1
    result = Tuple{ComplexF64,typeof(p1)}[];
    for k = 1:l
        a = prod((p1[kp,l+1]-p1[k,l]+k-kp+1 for kp in 1:l+1));
        b = l>1 ? prod((p1[kp,l-1]-p1[k,l]+k-kp for kp in 1:l-1)) : 1;
        c = prod((kp==k ? 1 : (p1[kp,l] - p1[k,l] + k - kp + 1)*(p1[kp,l]-p1[k,l]+k-kp) for kp in 1:l))

        pref = sqrt((-1.0+0im)*a*b/c)
        if pref != 0 && c != 0
            td = copy(p1.data);
            td[k,l]-=1;
            push!(result,(pref,GTPattern(td)));
        end
    end

    return result
end

#in the case of multiple fusion, we should gaugefix C
function gauge_fix(C)
    @warn "gauge fixing is not implemented yet"
    C
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
    invprodmap(z) = (z÷length(p2)+1,mod1(z,length(p2)));

    whw = Wz(p3[end]);
    N = p3[end].N

    T = fill(0.0+0im,length(p1)*length(p2),length(p1)*length(p2));
    used_dom = Vector{Tuple{Int64,Int64}}();
    used_codom = Vector{Tuple{Int64,Int64}}(); #this should be a set instead of calling unique on it at the end ...

    #get the allowed basis
    for (i,p1) in enumerate(p1),(j,p2) in enumerate(p2)
        if Wz(p1)+Wz(p2) == whw
            push!(used_dom,(i,j))
        end
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

    solutions = gauge_fix(nullspace(T_subslice));

    CGC = fill(0.0+0im,length(p3),size(solutions,2),length(p1),length(p2));
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

            B = fill(0.0+0im,length(parentbundle),length(children));
            T = fill(0.0+0im,length(parentbundle),size(CGC,2),length(p1),length(p2));
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

            infloop = false; # ironically, in my testing this introduced an infinite loop
            known[children].=true;
            delete!(weight2parent,k);
            break;
        end
    end
end
#actually calculating the CGC's
function CGC(s1::SUNIrrep{N},s2::SUNIrrep{N},p1 = GTpatterns(s1),p2 = GTpatterns(s2)) where N

    # step 1 : determine to which sectors s1 and s2 can fuse
    if length(p1) > length(p2)
        s3 = s2⊗s1
    else
        s3 = s1⊗s2
    end

    #of course, there is multiple fusion. We would like to know the outer multiplicity for every occuring irrep
    cp = countpartition(s3);
    canfuse = collect(keys(cp))
    map(canfuse) do irrep
        p3 = GTpatterns(irrep);

        # step 2 : determine the CGC of the largest weight GT pattern
        CGC = heighest_weight_CGC(p1,p2,p3);
        @assert size(CGC,2)>=cp[irrep]

        # step 3 : determine all other CGC's using ladder operators
        lower_weight_CGC!(CGC,p1,p2,p3)
        CGC
    end

end
