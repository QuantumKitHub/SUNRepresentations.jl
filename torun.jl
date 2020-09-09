using TensorKit,TensorOperations,MPSKit
import LinearAlgebra
matlab2gt(m,n) = SUNIrrep((m+n,n,0))

include("sector.jl")

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
