using Test
using TestExtras
using Random
using TensorKitSectors
using TensorKit
using SUNRepresentations
using Combinatorics
using TensorOperations
using Base.Iterators: take
using SparseArrayKit: SparseArray
using LinearAlgebra: LinearAlgebra

const TK = TensorKit

Random.seed!(1234)
Ti = time()

# sector tests
testsuite_path = joinpath(
    dirname(dirname(pathof(TensorKitSectors))), # TensorKitSectors root
    "test", "testsuite.jl"
)
include(testsuite_path)
using .SectorTestSuite: randsector, smallset

function SectorTestSuite.smallset(::Type{I}) where {I <: SUNIrrep}
    return smallset(I, 5, 15)
end
function SectorTestSuite.smallset(::Type{ProductSector{Tuple{I1, I2}}}) where {I1 <: SUNIrrep, I2 <: SUNIrrep}
    s1 = smallset(I1)
    s2 = smallset(I2)
    return resize!(shuffle!([a ⊠ b for a in s1 for b in s2 if dim(a) * dim(b) <= 500]), 5)
end

sectorlist = (SUNIrrep{3}, SUNIrrep{4}, SUNIrrep{5}, SUNIrrep{3} ⊠ SUNIrrep{3})
@testset "Sector tests" begin
    for sector in sectorlist
        SectorTestSuite.test_sector(sector)
    end
    include("sectors.jl")
end

module GenericTests
    using Test
    using TestExtras
    using Random
    using SUNRepresentations
    using TensorKitSectors
    include("generic.jl")
end

@testset "Caching tests" begin
    include("caching.jl")
end

sectorlist = (SUNIrrep{3}, SUNIrrep{4}, SUNIrrep{5})
include("fusiontrees.jl")

@testset "Aqua" verbose = true begin
    using Aqua
    # RationalRoots has ambiguities with Base/Core, so only test SUNRepresentations ambiguities
    # Intentional piracy of Rep[SU{N}] etc
    Aqua.test_all(SUNRepresentations; ambiguities = false, piracies = (; treat_as_own = [SU]))
    Aqua.test_ambiguities([SUNRepresentations])
end

Tf = time()
printstyled(
    "Finished all tests in ",
    string(round((Tf - Ti) / 60; sigdigits = 3)),
    " minutes."; bold = true, color = Base.info_color()
)
println()
