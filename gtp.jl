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

function highest_weight_GT(irr::SUNIrrep{N}) where N
    data = Matrix{Int64}(undef,N,N);
    for k = 1:N
        data[k,k:end].=irr.s[k];
        data[k,1:k-1].=0
    end
    GTPattern(data)
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
