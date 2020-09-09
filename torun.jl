using TensorKit,TensorOperations,MPSKit
import LinearAlgebra
matlab2gt(m,n) = SUNIrrep((m+n,n,0))
let

include("sector.jl")
#=
L1 = TensorMap(ones,ComplexF64,RepresentationSpace((SUNIrrep((0,0))=>1))*RepresentationSpace((SUNIrrep((2,0))=>1)),RepresentationSpace((SUNIrrep((2,0))=>1))*RepresentationSpace((SUNIrrep((2,0))=>1)));
R1 = TensorMap(ones,ComplexF64,RepresentationSpace((SUNIrrep((2,0))=>1))*RepresentationSpace((SUNIrrep((2,0))=>1)),RepresentationSpace((SUNIrrep((0,0))=>1))*RepresentationSpace((SUNIrrep((2,0))=>1)));
ham1 =  MPOHamiltonian([L1,R1]);

st1 = InfiniteMPS([RepresentationSpace((SUNIrrep((2,0))=>1))],[RepresentationSpace((SUNIrrep((1,0))=>4,SUNIrrep((3,0))=>4,SUNIrrep((5,0))=>4))]);
(gs1,_) = find_groundstate(st1,ham1,Vumps());
@show expectation_value(gs1,ham1);

L2 = TensorMap(ones,ComplexF64,ℂ[SU₂](0=>1)*ℂ[SU₂](1=>1),ℂ[SU₂](1=>1)*ℂ[SU₂](1=>1));
R2 = TensorMap(ones,ComplexF64,ℂ[SU₂](1=>1)*ℂ[SU₂](1=>1),ℂ[SU₂](0=>1)*ℂ[SU₂](1=>1));
ham2 =  MPOHamiltonian([L2,R2]);

st2 = InfiniteMPS([ℂ[SU₂](1=>1)],[ℂ[SU₂](1//2=>4,3//2=>4,5//2=>4)]);
(gs2,_) = find_groundstate(st2,ham2,Vumps());
@show expectation_value(gs2,ham2);
=#

a = TensorMap(rand,ComplexF64,RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(0,1) => 3)),RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(2,0) => 3)))
@show norm(a);
a = permute(a,(3,2,5),(1,4));
@show norm(a);

end
