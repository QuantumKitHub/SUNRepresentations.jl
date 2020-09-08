using TensorKit,TensorOperations; import LinearAlgebra; include("sector.jl")
matlab2gt(m,n) = SUNIrrep((m+n,n,0))
A = CGC(matlab2gt(2,2),matlab2gt(2,2),matlab2gt(2,2));

#=
using TensorKit,TensorOperations
import LinearAlgebra

include("sector.jl")

println("possible GT patterns of SU₂ spin 0 ")
@show GTpatterns(SUNIrrep((0,0)))

println("possible GT patterns of SU₂ spin 1/2")
@show GTpatterns(SUNIrrep((1,0)))

println("possible GT patterns of SU₂ spin 1")
@show GTpatterns(SUNIrrep((2,0)))

println("to what can we fuse SU₂ spin 1 and spin 2")
@show SUNIrrep((2,0))⊗SUNIrrep((4,0))

println("fusing spin 1/2 with spin 1 to spin 3/2")
CGC(SUNIrrep((1,0)),SUNIrrep((2,0)),SUNIrrep((3,0)))

println("SU(3) is no problem")
coeff = CGC(SUNIrrep((2,1,0)),SUNIrrep((2,1,0)),SUNIrrep((3, 3, 0)));
@tensor test[-1 -2;-3 -4]:=coeff[-1,-2,1,2]*conj(coeff[-3,-4,1,2])
D = size(test,1)*size(test,2);
isunitary = norm(reshape(test,D,D)-Matrix(LinearAlgebra.I,D,D))
println("unitary up to $(isunitary)")


# this works - but only on a local copy
using TensorKit,MPSKit


L = TensorMap(ones,ComplexF64,RepresentationSpace((SUNIrrep((0,0))=>1))*RepresentationSpace((SUNIrrep((2,0))=>1)),RepresentationSpace((SUNIrrep((2,0))=>1))*RepresentationSpace((SUNIrrep((2,0))=>1)));
R = TensorMap(ones,ComplexF64,RepresentationSpace((SUNIrrep((2,0))=>1))*RepresentationSpace((SUNIrrep((2,0))=>1)),RepresentationSpace((SUNIrrep((0,0))=>1))*RepresentationSpace((SUNIrrep((2,0))=>1)));
ham =  MPOHamiltonian([L,R]);

st = InfiniteMPS([RepresentationSpace((SUNIrrep((2,0))=>1))],[RepresentationSpace((SUNIrrep((1,0))=>4,SUNIrrep((3,0))=>4,SUNIrrep((5,0))=>4))]);
(gs,pars,_) = find_groundstate(st,ham,Vumps());
@show expectation_value(gs,ham);

L = TensorMap(ones,ComplexF64,ℂ[SU₂](0=>1)*ℂ[SU₂](1=>1),ℂ[SU₂](1=>1)*ℂ[SU₂](1=>1));
R = TensorMap(ones,ComplexF64,ℂ[SU₂](1=>1)*ℂ[SU₂](1=>1),ℂ[SU₂](0=>1)*ℂ[SU₂](1=>1));
ham =  MPOHamiltonian([L,R]);

st = InfiniteMPS([ℂ[SU₂](1=>1)],[ℂ[SU₂](1//2=>4,3//2=>4,5//2=>4)]);
(gs,pars,_) = find_groundstate(st,ham,Vumps());
@show expectation_value(gs,ham);
=#
