using RowEchelon

#=
GTP = gelfand-Tsetlin pattern

sectors.jl contains the sector definition
gtp.jl contains gtp bells and whistles
cgc.jl uses gtp/sectors to calculate clebschgordans

things to change
    - data container in GTPattern
    - some names
    - if I implement the indexing from the paper, the CGC routine will be so much nicer (no more findindex)
    - use sparse tensors for the CGCs

However, runtime should be in the matrix inversion / nullspace subroutines anyway
=#

struct SUNIrrep{N}<:Irrep{SU{N}}
    s::NTuple{N,Int64}
end
SUNIrrep{N}(i::Vararg{Int64,N}) where N= SUNIrrep(i);

function Base.isless(s1::SUNIrrep{N},s2::SUNIrrep{N}) where N
    n_s1 = normalize(s1);
    n_s2 = normalize(s2);

    return isless(n_s1.s,n_s2.s)
end

Base.IteratorSize(::Type{TensorKit.SectorValues{T}}) where T<:SUNIrrep = Base.IsInfinite()


function Base.iterate(::TensorKit.SectorValues{T}, i = ntuple(x->0,Val{N}()))  where T<:SUNIrrep{N} where N
    ii = 1;
    for j = 2:N-1
        ii = i[j-1]>=i[j]+1>=i[j+1] ? j : ii
    end
    ni = i.+ntuple(x->x==ii ? 1 : 0,Val{N}());
    (SUNIrrep(i),ni)
end

Base.isequal(s::SUNIrrep{N},t::SUNIrrep{N}) where N = isequal(s.s,t.s);
Base.hash(s::SUNIrrep,h::UInt) = hash(s.s,h);
function TensorKit.dim(s::SUNIrrep{N}) where N
    #Int(prod((prod((1+(s.s[k1]-s.s[k2])//(k2-k1) for k1 = 1:k2-1)) for k2 = 2:N))) # original which doesn't infer correctly
    toret::Rational{Int64} = 1//1;
    for k2 = 2:N,k1 = 1:k2-1
        toret*=(k2-k1+s.s[k1]-s.s[k2])//(k2-k1)
    end
    @assert denominator(toret) == 1;
    return numerator(toret)
end
TensorKit.normalize(s::SUNIrrep) = SUNIrrep(s.s.-s.s[end]);

Base.conj(s::SUNIrrep) = SUNIrrep((reverse(s.s).-s.s[1]).*-1) #maybe? https://aip.scitation.org/doi/pdf/10.1063/1.1704095
Base.one(::Type{SUNIrrep{N}}) where N = SUNIrrep(ntuple(x->0,N));


Base.@pure TensorKit.FusionStyle(::Type{SUNIrrep{N}}) where N = DegenerateNonAbelian()
Base.isreal(::Type{SUNIrrep{N}}) where N = true
TensorKit.BraidingStyle(::Type{SUNIrrep{N}}) where N = Bosonic();


TensorKit.:⊗(s1::SUNIrrep{N},s2::SUNIrrep{N}) where N = unique(_otimes(s1,s2))
TensorKit.Nsymbol(s1::SUNIrrep{N},s2::SUNIrrep{N},s3::SUNIrrep{N}) where N = count(x->x==s3,_otimes(s1,s2));

function TensorKit.Fsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}, d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where N
    N1 = Nsymbol(a,b,e)
    N2 = Nsymbol(e,c,d)
    N3 = Nsymbol(b,c,f)
    N4 = Nsymbol(a,f,d)

    (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0) &&
        return fill(0.0,max(1,N1),max(1,N2),max(1,N3),max(1,N4))
    A = CGC(a,b,e)
    B = CGC(e,c,d)
    C = CGC(b,c,f)
    D = CGC(a,f,d)

    @tensor F[-1 -2 -3 -4] := conj(A[3,-1,1,2])*conj(B[6,-2,3,4])*C[5,-3,2,4]*D[6,-4,1,5]
    Array(F)[1:N1,1:N2,1:N3,1:N4]::Array{Float64,4}/dim(d)
end

function TensorKit.Rsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where N
    N1 = Nsymbol(a,b,c);
    N2 = Nsymbol(b,a,c);

    (N1 == 0 || N2 == 0) && return fill(0.0,max(1,N1),max(1,N2));
    A = CGC(a,b,c)
    B = CGC(b,a,c)

    @tensor R[-1;-2] := A[3,-1,1,2]*conj(B[3,-2,2,1])
    Array(R)[1:N1,1:N2]/dim(c)
end

#this is not the correct \otimes, it repeats in case of multiplicities
function _otimes(s1::SUNIrrep{N},s2::SUNIrrep{N}) where N
    dim(s1)>dim(s2) && return _otimes(s2,s1);

    result = SUNIrrep{N}[];

    for pat in GTpatterns(s1)
        t = s2.s;
        bad_patern = false;
        for k in 1:N,l=N:-1:k
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

        if !(bad_patern)
            push!(result,SUNIrrep(t))
        end
    end

    sort(normalize.(result));
end

function TensorKit.fusiontensor(s1::SUNIrrep{N},s2::SUNIrrep{N},s3::SUNIrrep{N}) where N
    ft = permutedims(CGC(s1,s2,s3),(3,4,1,2))[:,:,:,1:Nsymbol(s1,s2,s3)];
end

include("gtp.jl")
include("cgc.jl")
