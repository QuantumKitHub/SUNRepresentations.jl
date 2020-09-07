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
isunitary = norm(reshape(test,D,D)-Matrix(I,D,D))
println("unitary up to $(isunitary)")
