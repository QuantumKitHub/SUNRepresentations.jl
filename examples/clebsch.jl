#=
an example on how to calculate the clebsch gordon coefficients for some su3 irreps
=#
using SUNRepresentations;

a = SUNIrrep(2,1,0);
b = SUNIrrep(2,1,0);
c = SUNIrrep(2,1,0);

CGC(a,b,c)
