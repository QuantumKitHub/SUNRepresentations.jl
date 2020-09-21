function qrpos(C)
    (q,r) = LinearAlgebra.qr(C);
    D = LinearAlgebra.diagm(sign.(LinearAlgebra.diag(q)));
    (q*D,D*r)
end
