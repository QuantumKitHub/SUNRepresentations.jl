using TensorKit,SUNRepresentations
matlab2gt(m,n) = SUNRepresentations.SUNIrrep((m+n,n,0))

a = TensorMap(rand,ComplexF64,RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(0,1) => 3)),RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(2,0) => 3)))
@show norm(a);
#a = permute(a,(3,2,5),(1,4));
#a = permute(a,(4,2,1),(5,3));
a = permute(a,(1,),(2,3,4,5));
a = permute(a,(1,2,3),(4,5))
@show norm(a);
