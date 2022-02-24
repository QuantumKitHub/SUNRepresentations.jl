#=
an example on how to calculate the clebsch gordon coefficients for some su3 irreps
=#
using SUNRepresentations;

a = SUNIrrep(2,1,0);
b = SUNIrrep(2,1,0);
c = SUNIrrep(2,1,0);

CGC(a,b,c)

let

d = SUNRepresentations.CGC(Irrep(2,1,0),Irrep(2,1,0),Irrep(2,1,0));
@tensor temp[-1 -2;-3 -4] := d[1 2 -1 -2]*conj(d[1 2 -3 -4])

end
