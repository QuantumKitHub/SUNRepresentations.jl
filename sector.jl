using LinearAlgebra,TensorOperations,TensorOperations,TensorKit
#=
Base.IteratorSize(::Type{SectorValues{SU2Irrep}}) = IsInfinite()
Base.iterate(::SectorValues{SU2Irrep}, i = 0) = (SU2Irrep(half(i)), i+1)
Base.getindex(::SectorValues{SU2Irrep}, i::Int) =
    1 <= i ? SU2Irrep(half(i-1)) : throw(BoundsError(values(SU2Irrep), i))
findindex(::SectorValues{SU2Irrep}, s::SU2Irrep) = twice(s.j)+1

# SU2Irrep(j::Real) = convert(SU2Irrep, j)
Base.convert(::Type{SU2Irrep}, j::Real) = SU2Irrep(j)

Base.@pure FusionStyle(::Type{SU2Irrep}) = SimpleNonAbelian()
Base.isreal(::Type{SU2Irrep}) = true

Nsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep) = WignerSymbols.δ(sa.j, sb.j, sc.j)
function Fsymbol(s1::SU2Irrep, s2::SU2Irrep, s3::SU2Irrep,
                    s4::SU2Irrep, s5::SU2Irrep, s6::SU2Irrep)
    if all(==(_su2one), (s1, s2, s3, s4, s5, s6))
        return 1.0
    else
        return sqrt(dim(s5) * dim(s6)) * WignerSymbols.racahW(Float64, s1.j, s2.j,
                                                                s4.j, s3.j, s5.j, s6.j)
    end
end
function Rsymbol(sa::SU2Irrep, sb::SU2Irrep, sc::SU2Irrep)
    Nsymbol(sa, sb, sc) || return 0.
    iseven(convert(Int, sa.j+sb.j-sc.j)) ? 1.0 : -1.0
end

function fusiontensor(a::SU2Irrep, b::SU2Irrep, c::SU2Irrep, v::Nothing = nothing)
    C = Array{Float64}(undef, dim(a), dim(b), dim(c))
    ja, jb, jc = a.j, b.j, c.j

    for kc = 1:dim(c), kb = 1:dim(b), ka = 1:dim(a)
        C[ka,kb,kc] = WignerSymbols.clebschgordan(ja, ja+1-ka, jb, jb+1-kb, jc, jc+1-kc)
    end
    return C
end

Base.hash(s::SU2Irrep, h::UInt) = hash(s.j, h)
Base.isless(s1::SU2Irrep, s2::SU2Irrep) = isless(s1.j, s2.j)

=#
struct SUNIrrep{N}
    s::NTuple{N,Int64}
end

TensorKit.dim(s::SUNIrrep{N}) where N= prod((prod(((s.s[k1]-s.s[k2])/(k1-k2) for k1 = 1:k2-1)) for k2 = 2:N))
TensorKit.normalize(s::SUNIrrep) = SUNIrrep(s.s.-s.s[end]);
Base.conj(s::SUNIrrep) = s;
Base.one(s::SUNIrrep{N}) where N = SUNIrrep(ntuple(x->0,N));

function TensorKit.:⊗(s1::SUNIrrep{N},s2::SUNIrrep{N}) where N
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

    normalize.(result);
end
include("gtp.jl")
