include("gtp.jl")

println("possible GT patterns of SU₂ spin 0 ")
@show GT_patterns((0,0))

println("possible GT patterns of SU₂ spin 1/2")
@show GT_patterns((1,0))

println("possible GT patterns of SU₂ spin 1")
@show GT_patterns((2,0))

println("fusing spin 1/2 with spin 1")
CGC((1,0),(2,0))
