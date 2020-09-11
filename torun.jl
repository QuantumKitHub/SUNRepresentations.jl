using TensorKit,MPSKit
import LinearAlgebra
include("sector.jl")
matlab2gt(m,n) = SUNIrrep((m+n,n,0))
let

#=
a = TensorMap(rand,ComplexF64,RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(0,1) => 3)),RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(2,0) => 3)))
@show norm(a);
a = permute(a,(3,2,5),(1,4));
@show norm(a);
=#

(v1,v2,v3,v4,v5) = (ℂ[SUNIrrep{3}](SUNIrrep{3}(0,0,0)=>2,SUNIrrep{3}(1,0,0)=>2,SUNIrrep{3}(1,1,0)=>2),
        ℂ[SUNIrrep{3}](SUNIrrep{3}(0,0,0)=>1,SUNIrrep{3}(2,0,0)=>2),
        ℂ[SUNIrrep{3}](SUNIrrep{3}(0,0,0)=>2,SUNIrrep{3}(2,0,0)=>3,SUNIrrep{3}(2,1,0)=>5),
        ℂ[SUNIrrep{3}](SUNIrrep{3}(1,0,0)=>2,SUNIrrep{3}(3,0,0)=>2),
        ℂ[SUNIrrep{3}](SUNIrrep{3}(2,1,0)=>2,SUNIrrep{3}(1,1,0)=>2,SUNIrrep{3}(3,1,0)=>2));

t = TensorMap(rand,ComplexF64,v1*v2*v3,v4*v5)
for (f1,f2) in fusiontrees(t)
    t[f1,f2]
end
end
