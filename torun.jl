using TensorKit,MPSKit,TensorOperations
import LinearAlgebra
include("sector.jl")
matlab2gt(m,n) = SUNIrrep((m+n,n,0))
let

d = CGC(SUNIrrep((1,0,0)),SUNIrrep((1,0,0)),SUNIrrep((2,0,0)));
@tensor temp[-1 -2;-3 -4] := d[-1 -2 1 2]*conj(d[-3 -4 1 2])
@show temp
#=
a = TensorMap(rand,ComplexF64,RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(0,1) => 3)),RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(2,0) => 3)))
@show norm(a);
a = permute(a,(3,2,5),(1,4));
@show norm(a);
=#
end
