using TensorKit,TensorOperations,MPSKit
import LinearAlgebra
matlab2gt(m,n) = SUNIrrep((m+n,n,0))
let

include("sector.jl")

out1 = (SUNIrrep{3}((1, 0, 0)), SUNIrrep{3}((1, 1, 0)))
in1 = SUNIrrep{3}((0, 0, 0))
out2 = (SUNIrrep{3}((1, 1, 0)), SUNIrrep{3}((1, 0, 0)), SUNIrrep{3}((1, 1, 0)))
in2 = SUNIrrep{3}((2, 0, 0))
for f1 = fusiontrees((out1..., dual(in1)), one(in1)),f2 = fusiontrees(out2, in2)
    for c in f1.coupled ⊗ f2.coupled
        FusionTree((f1.coupled, f2.coupled), c, (false, false), ())
    end
end
end
