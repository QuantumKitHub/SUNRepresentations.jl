using TensorOperations,SUNRepresentations;

let

d = SUNRepresentations.CGC(Irrep(2,1,0),Irrep(2,1,0),Irrep(2,1,0));
@tensor temp[-1 -2;-3 -4] := d[1 2 -1 -2]*conj(d[1 2 -3 -4])

end
