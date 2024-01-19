
using Test
using TestExtras
using Aqua
using Random
using TensorKit
using SUNRepresentations
using Combinatorics
using TensorKit
using TensorKit: ProductSector, fusiontensor, pentagon_equation, hexagon_equation
using TensorOperations
using Base.Iterators: take, product
using SparseArrayKit: SparseArray
using LinearAlgebra: LinearAlgebra

const TK = TensorKit

Random.seed!(1234)

smallset(::Type{I}) where {I<:Sector} = take(values(I), 5)
function smallset(::Type{ProductSector{Tuple{I1,I2}}}) where {I1,I2}
    iter = product(smallset(I1), smallset(I2))
    s = collect(i ⊠ j for (i, j) in iter if dim(i) * dim(j) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function smallset(::Type{ProductSector{Tuple{I1,I2,I3}}}) where {I1,I2,I3}
    iter = product(smallset(I1), smallset(I2), smallset(I3))
    s = collect(i ⊠ j ⊠ k for (i, j, k) in iter if dim(i) * dim(j) * dim(k) <= 6)
    return length(s) > 6 ? rand(s, 6) : s
end
function randsector(::Type{I}) where {I<:Sector}
    s = collect(smallset(I))
    a = rand(s)
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end

Ti = time()
module GenericTests
using Test
using TestExtras
using Random
using SUNRepresentations
include("generic.jl")
end

include("caching.jl")
sectorlist = (SUNIrrep{3}, SUNIrrep{4}, SUNIrrep{5}, SUNIrrep{3} ⊠ SUNIrrep{3})
include("sectors.jl")
sectorlist = (SUNIrrep{3}, SUNIrrep{4}, SUNIrrep{5})
include("fusiontrees.jl")

@testset "Aqua" verbose = true begin
    # RationalRoots has ambiguities with Base/Core, so only test SUNRepresentations ambiguities
    # Intentional piracy of Rep[SU{N}] etc
    Aqua.test_all(SUNRepresentations; ambiguities=false, piracies=(; treat_as_own=[SU]))
    Aqua.test_ambiguities([SUNRepresentations])
end

Tf = time()
printstyled("Finished all tests in ",
            string(round((Tf - Ti) / 60; sigdigits=3)),
            " minutes."; bold=true, color=Base.info_color())
println()
