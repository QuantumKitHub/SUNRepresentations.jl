module SUNRepresentationsLatexifyExt

using SUNRepresentations
using SUNRepresentations: dimname, parse_dimname
using Latexify
using Latexify: @latexrecipe, LaTeXString

@latexrecipe function f(x::SUNIrrep)
    ## set parameters
    env --> :inline
    
    ## convert into latex string
    d, numprimes, conjugate = parse_dimname(dimname(x))
    str_new = conjugate ? "\\overline{\\textbf{$d}}" : "\\textbf{$d}"
    if numprimes == 1
        str_new *= "^\\prime"
    elseif numprimes > 1
        str_new *= "^{" * repeat("\\prime", numprimes) * "}"
    end
    return LaTeXString(str_new)
end

Base.show(io::IO, ::MIME"text/latex", x::SUNIrrep) = print(io, latexify(x))

end