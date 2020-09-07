using LinearAlgebra,TensorOperations,TensorOperations,TensorKit

struct SUNIrrep{N}<:Sector
    s::NTuple{N,Int64}
end


function Base.isless(s1::SUNIrrep{N},s2::SUNIrrep{N}) where N
    n_s1 = normalize(s1);
    n_s2 = normalize(s2);

    iseq = true;
    for n in 1:N
        n_s1.s[n]>n_s2.s[n] && return false
        iseq = iseq && (n_s1.s[n]==n_s2.s[n])
    end
    return !iseq
end

Base.IteratorSize(::Type{TensorKit.SectorValues{T}}) where T<:SUNIrrep = Base.IsInfinite()

Base.isequal(s::SUNIrrep{N},t::SUNIrrep{N}) where N = isequal(s.s,t.s);
Base.hash(s::SUNIrrep,h::UInt) = hash(s.s,h);
TensorKit.dim(s::SUNIrrep{N}) where N= prod((prod(((s.s[k1]-s.s[k2])/(k1-k2) for k1 = 1:k2-1)) for k2 = 2:N))
TensorKit.normalize(s::SUNIrrep) = SUNIrrep(s.s.-s.s[end]);

Base.conj(s::SUNIrrep) = s;
Base.one(::Type{SUNIrrep{N}}) where N = SUNIrrep(ntuple(x->0,N));


#=Base.@pure =#TensorKit.FusionStyle(::Type{SUNIrrep{N}}) where N = DegenerateNonAbelian()
Base.isreal(::Type{SUNIrrep{N}}) where N = false #not sure - but complex is anyway more general
TensorKit.BraidingStyle(::Type{SUNIrrep{N}}) where N = Bosonic();


TensorKit.:⊗(s1::SUNIrrep{N},s2::SUNIrrep{N}) where N = unique(_otimes(s1,s2))
TensorKit.Nsymbol(s1::SUNIrrep{N},s2::SUNIrrep{N},s3::SUNIrrep{N}) where N = count(x->x==s3,_otimes(s1,s2));

function TensorKit.Fsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}, d::SUNIrrep{N}, e::SUNIrrep{N}, f::SUNIrrep{N}) where N
    A = CGC(a,b,e)
    B = CGC(e,c,d)
    C = CGC(b,c,f)
    D = CGC(a,f,d)

    @tensor conj(A[1,2,3])*conj(B[3,4,-2])*C[2,4,5]*D[1,5,-1]
end

function TensorKit.Rsymbol(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where N
    @tensor CGC(b,a,c)[1,2,-2]*conj(CGC(a,b,c)[2,1,-1])
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

include("gtp.jl")
