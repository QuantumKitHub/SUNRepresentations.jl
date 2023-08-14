#=
example that constructs an su3 symmetric tensor, and does a contraction
=#
using TensorKit, SUNRepresentations

t = TensorMap(rand, ComplexF64,
              Rep[SU{3}]((2, 1, 0) => 3) * Rep[SU{3}]((2, 1, 0) => 3) *
              Rep[SU{3}]((2, 0, 0) => 1),
              Rep[SU{3}]((2, 1, 0) => 3) * Rep[SU{3}]((2, 0, 0) => 3));
@tensor v[-1; -2] := t[-1 1 2; 3 4] * conj(t[-2 1 2; 3 4])
