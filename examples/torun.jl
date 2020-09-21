using TensorOperations,SUNClebschGordan;
matlab2gt(m,n) = SUNIrrep((m+n,n,0))
let

d = CGC(SUNIrrep((2,1,0)),SUNIrrep((2,1,0)),SUNIrrep((2,1,0)));
@tensor temp[-1 -2;-3 -4] := d[-1 -2 1 2]*conj(d[-3 -4 1 2])
@show reshape(Array(temp),size(temp,1)*size(temp,2),size(temp,1)*size(temp,2))

#=
a = TensorMap(rand,ComplexF64,RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(0,1) => 3)),RepresentationSpace((matlab2gt(1,1) => 3))*RepresentationSpace((matlab2gt(2,0) => 3)))
@show norm(a);
a = permute(a,(3,2,5),(1,4));
@show norm(a);
a = permute(a,(4,2,1),(5,3));
@show norm(a);
a = permute(a,(3,2,5),(1,4));
@show norm(a);
a = permute(a,(4,2,1),(5,3));
@show norm(a);
=#
end
