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

#find all gtpatterns associated to a given irrep
function GTpatterns(irr::SUNIrrep{N}) where N
    N == 1 && return [GTPattern(fill(irr.s[1],1,1))]


    patterns = reduce(vcat,map(Iterators.product(ntuple(i->irr.s[i+1]:irr.s[i],N-1)...)) do x
        GTpatterns(SUNIrrep(x))
    end)::Vector{GTPattern}

    patterns = map(patterns) do pat
        #we have the paterns from a descendent of irr, so now we insert irr itself
        GTPattern(hcat(vcat(pat.data,fill(0,1,N-1)),collect(irr.s)))
    end

    sort(patterns)
end

function creation(ps::Vector{GTPattern})
    N = ps[1].N;
    result = [SparseArray{Float64}(undef,length(ps),length(ps)) for i in 1:N-1];

    for (i,p) in enumerate(ps)
        for l = 1:N-1,k = 1:l
            pref = -1//1;
            for kp in 1:l+1
                pref*=p[kp,l+1]-p[k,l]+k-kp;
                if kp <= l-1; pref*=p[kp,l-1]-p[k,l]+k-kp-1; end
                numerator(pref) == 0 && break;

                if kp <= l && kp != k; pref//= (p[kp,l] - p[k,l] + k - kp)*(p[kp,l]-p[k,l]+k-kp-1);end
                denominator(pref) == 0 && break;
            end

            (denominator(pref) == 0 || numerator(pref) == 0) && continue;

            td = deepcopy(p.data); td[k,l]+=1;
            fi = findfirst(x->isequal(x.data,td),ps);
            result[l][fi,i] += sqrt(pref);
        end
    end

    return result;
end

function anihilation(ps::Vector{GTPattern})
    N = ps[1].N;
    result = [SparseArray{Float64}(undef,length(ps),length(ps)) for i in 1:N-1];

    for (i,p) in enumerate(ps)
        for l = 1:N-1,k = 1:l
            pref = -1//1;
            for kp in 1:l+1
                pref*=p[kp,l+1]-p[k,l]+k-kp+1;
                if kp <= l-1; pref*=p[kp,l-1]-p[k,l]+k-kp; end
                numerator(pref) == 0 && break;

                if kp <= l && kp != k; pref//= (p[kp,l] - p[k,l] + k - kp + 1)*(p[kp,l]-p[k,l]+k-kp);end
                denominator(pref) == 0 && break;
            end

            (denominator(pref) == 0 || numerator(pref) == 0) && continue;

            td = deepcopy(p.data); td[k,l]-=1;
            fi = findfirst(x->isequal(x.data,td),ps);
            result[l][fi,i] += sqrt(pref);
        end
    end

    return result;
end
