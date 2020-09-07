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

a = TensorMap(rand,ComplexF64,RepresentationSpace((SUNIrrep((2,0))=>2)),RepresentationSpace((SUNIrrep((2,0))=>2)))
a = permute(a,(2,),(1,));
a = permute(a,(1,2),(,));

a = TensorMap(rand,ComplexF64,RepresentationSpace((SUNIrrep((2,0))=>2,SUNIrrep((2,0))=>2)),RepresentationSpace((SUNIrrep((4,0))=>2)))
