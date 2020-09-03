#=
This file contains the actual implementation of the paper

GTP = gelfand-Tsetlin pattern
irreps are tuples


things to change
    - data container in GTPattern
    - some names
    - if I implement the indexing from the paper, the CGC routine will be so much nicer (no more findindex)
    - break down some routines in subroutines

However, runtime should be in the matrix inversion / nullspace subroutines anyway

currently errors for some reason (lapack exception) when determining nullspace for large things
=#

using LinearAlgebra,TensorOperations

struct GTPattern
    data::Matrix{Int64}
end
Base.getproperty(d::GTPattern,s::Symbol) = s == :N ? size(d.data,1) : getfield(d,s);

Base.getindex(d::GTPattern,i,j) = d.data[i,j]
Base.checkbounds(d::GTPattern,i,j) = i>0 && i <=j && j>0 && j<=d.N
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
Base.isequal(a::GTPattern,b::GTPattern) = hash(a)==hash(b)
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
function GT_pattern_down(irr::NTuple{N,Int64}) where N
    @assert N>1

    toret = collect(Iterators.product(ntuple(i->irr[i+1]:irr[i],N-1)...))
    toret[:]
end

function GT_patterns(irr::NTuple{N,Int64}) where N
    N == 1 && return [GTPattern(fill(irr[1],1,1))]

    patterns::Vector{GTPattern} = reduce(vcat,GT_patterns.(GT_pattern_down(irr)));

    patterns = map(patterns) do pat
        #we have the paterns from a descendent of irr, so now we insert irr itself
        GTPattern(hcat(vcat(pat.data,fill(0,1,N-1)),collect(irr)))
    end

    sort(patterns)
end

function highest_weight_GT(irr::NTuple{N,Int64}) where N
    data = Matrix{Int64}(undef,N,N);
    for k = 1:N
        data[k,k:end].=irr[k];
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

standardize(s1::NTuple{N,Int64}) where N = s1.-s1[end];

function whichparts(s1::NTuple{N,Int64},s2::NTuple{N,Int64},p1=GT_patterns(s1)) where N
    result = NTuple{N,Int64}[];

    for pat in p1
        t = s2;
        bad_patern = false;
        for k in 1:N
            for l = N:-1:k
                bad_patern && break;

                bkl = pat[k,l]


                if checkbounds(pat,k,l-1)
                    bkl -= pat[k,l-1]
                end


                t = Base.setindex(t,t[l]+bkl,l)

                if l>1
                    bad_patern = t[l-1]< t[l]
                end
            end
        end

        if !(bad_patern)
            push!(result,t)
        end
    end

    standardize.(result);
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

#returns weight2child and weight2parent maps
function weightmaps(w3)

    weight2child = Dict{eltype(w3),Vector{Int}}()
    for (i,w) in enumerate(w3)
        if !(w in keys(weight2child))
            weight2child[w] = Int[];
        end
        push!(weight2child[w],i)
    end

    weight2parent = Dict{eltype(w3),Vector{Tuple{Int,Int}}}();
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

    return weight2child,weight2parent
end

#actually calculating the CGC's
function CGC(s1::NTuple{N,Int64},s2::NTuple{N,Int64},p1 = GT_patterns(s1),p2 = GT_patterns(s2)) where N
    w1 = Wz.(p1);
    w2 = Wz.(p2);

    println("determining which sectors $(s1) and $(s2) fuse to")
    if length(p1) > length(p2)
        s3 = whichparts(s2,s1,p2);
    else
        s3 = whichparts(s1,s2,p1);
    end

    #of course, there is multiple fusion. We would like to know the outer multiplicity for every occuring irrep
    cp = countpartition(s3);
    for (irrep,NSSS #=borowing names from the paper=#) in cp
        println("--------------------------------------------")
        println("fusing to $(irrep) with multiplicity $(NSSS)")

        p3 = GT_patterns(irrep);
        hw = p3[end]
        whw = Wz(hw);

        println("the heighest weight GT pattern (wz = $(whw)): ")
        @show hw

        prodmap(t) = prodmap(t...);
        prodmap(i,j) = (i-1)*length(p2)+j
        invprodmap(z) = (z÷length(p2)+1,mod1(z,length(p2)));

        T = Matrix{ComplexF64}(undef,length(p1)*length(p2),length(p1)*length(p2));
        used_dom = Vector{Tuple{Int64,Int64}}();

        #get the allowed basis
        for i in 1:length(p1),j in 1:length(p2)
            if w1[i]+w2[j] == whw
                push!(used_dom,(i,j))
            end
        end

        #construct 'T'; we then solve C*T = 0
        used_codom = Vector{Tuple{Int64,Int64}}();
        for (j,k) in used_dom
            for l in 1:N-1
                for (pref,ap1) in creation(p1[j],l)
                    x = (findfirst(x->isequal(x,ap1),p1),k);
                    push!(used_codom,x)
                    T[prodmap(x),prodmap(j,k)] = pref;
                end
                for (pref,ap2) in creation(p2[k],l)
                    x = (j,findfirst(x->isequal(x,ap2),p2));
                    push!(used_codom,x)
                    T[prodmap(x),prodmap(j,k)] = pref;
                end
            end
        end

        solutions = gauge_fix(nullspace(T[prodmap.(used_codom),prodmap.(used_dom)]));

        #println("is solutions unitary?")
        #@show solutions'*solutions
        #ja tis unitair ....

        @assert size(solutions,2)>=NSSS

        CGC = fill(0.0+0im,length(p3),size(solutions,2),length(p1),length(p2));
        for α in 1:size(solutions,2)
            for (i,(j,k)) in enumerate(used_dom)
                CGC[end#=highest weight=#,α,j,k] = solutions[i,α]
            end
        end

        println("found CGC to heighest weight GTP");

        #at this point, we have the CGC's for the highest weight GT pattern
        #now we use ladder operators to construct the other CGCs
        wp3 = W.(p3);
        (weight2child,weight2parent) = weightmaps(wp3);
        known = fill(false,length(p3)); known[end] = true;

        @assert isempty(weight2parent[wp3[end]]) # there are no parents of the largest weight irrep
        delete!(weight2parent,wp3[end]);

        println("determining lower weights GTPs")
        infloop = false
        while !infloop
            infloop = true;

            for (k,parentbundle) in weight2parent
                !reduce(&,map(x->known[x[1]],parentbundle)) && continue

                println("-----")
                println("know all parents to weight $k : ")

                for cv in parentbundle; @show p3[cv[1]];end;

                println("children:")
                children = weight2child[k];
                for cv in children; @show p3[cv]; end;

                B = fill(0.0+0im,length(parentbundle),length(children));
                T = fill(0.0+0im,length(parentbundle),size(solutions,2),length(p1),length(p2));
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

        println("found all CGC to this irrep")
        @show CGC

        println("is it unitary?")
        @tensor derp[-1 -2;-3 -4]:=CGC[-1,-2,1,2]*conj(CGC[-3,-4,1,2])
        @show derp
    end

end
