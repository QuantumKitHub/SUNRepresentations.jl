include("sector.jl")

println("possible GT patterns of SU₂ spin 0 ")
@show GTpatterns(SUNIrrep((0,0)))

println("possible GT patterns of SU₂ spin 1/2")
@show GTpatterns(SUNIrrep((1,0)))

println("possible GT patterns of SU₂ spin 1")
@show GTpatterns(SUNIrrep((2,0)))

println("to what can we fuse SU₂ spin 1 and spin 2")
@show SUNIrrep((2,0))⊗SUNIrrep((4,0))

println("fusing spin 1/2 with spin 1")
CGC(SUNIrrep((1,0)),SUNIrrep((2,0)))
