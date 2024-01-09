# SUNRepresentations.jl

[![Build
Status](https://github.com/maartenvd/SUNRepresentations.jl/workflows/CI/badge.svg)](https://github.com/maartenvd/SUNRepresentations.jl/actions)
[![Coverage](https://codecov.io/gh/maartenvd/SUNRepresentations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/maartenvd/SUNRepresentations.jl/branch/master)

Compute Clebsch-Gordan coefficients for general SU(N) groups. Reimplementation of [arXiv:1009.0437](https://arxiv.org/pdf/1009.0437.pdf). Compatibility / interoperability with [TensorKit.jl](https://github.com/Jutho/TensorKit.jl).

## Installation

```julia-repl
julia> using Pkg; Pkg.add("SUNRepresentations")
```

## Usage

```julia
using TensorKit, SUNRepresentations
I = SUNIrrep(2, 1, 0)
println("$I ⊗ $I = $(collect(I ⊗ I))")
```

```
Irrep[SU{3}]((2, 1, 0)) ⊗ Irrep[SU{3}]((2, 1, 0)) = SUNIrrep{3}[(0, 0, 0), (4, 2, 0), (3, 3, 0), (2, 1, 0), (3, 0, 0)]
```

## Caching Clebsch-Gordan coefficients

Because the computation of Clebsch-Gordan coefficients can be quite expensive, they are
cached to a file on disk, while a limited amount is kept in the RAM. It is possible to
precompute a set of Clebsch-Gordan coefficients and store them on disk, so that they can be
reused later. This is especially useful when computing Clebsch-Gordan coefficients in
parallel. This can be done using the `SUNRepresentations.precompute_disk_cache(N, a_max)`
function.

By default, the coefficients are only cached in memory, and will
be lost when the Julia session is closed. In order to cache the coefficients that are
currently stored in memory to disk, you can call `SUNRepresentations.sync_disk_cache(N)`.
This will create a scratchfile, which will then automatically be used the next time you want
to compute Clebsch-Gordan coefficients. Often, it can be useful to precompute a large set of
Clebsch-Gordan coefficients (in parallel) and store them on disk. This can be done using the
`SUNRepresentations.precompute_disk_cache(N, a_max)` function. This has the additional
benefit that the coefficients can be computed in parallel.

```julia-repl
julia> SUNRepresentations.precompute_disk_cache(3, 2)
CGC RAM cache info:
===================

CGC disk cache info:
====================
SU(3) - Float64
    * 65 entries
    * 135.191 KiB
```

They are stored at `SUNRepresentations.cache_path(N)` and can be removed using
`SUNRepresentations.clear_disk_cache([N])`. To display information, use
`SUNRepresentations.cache_info()`.

## Conventions

By default, irreps are denoted by their `N` weights, which are equivalent to the number of
boxes in each row of the Young tableau, and this is also how they are stored. For example,
the fundamental representation of SU(3) is denoted by `SUNIrrep(1, 0, 0)`, and the adjoint
representation by `SUNIrrep(2, 1, 0)`. Nevertheless, we also support using `N - 1` Dynkin
labels, which are denoted using `Vector{Int}`. For example, the fundamental representation
of SU(3) is denoted by `SUNIrrep([1, 0])`, and the adjoint representation by
`SUNIrrep([1, 1])`. Finally, it is also possible to use the dimensional name which is often
used in physics, e.g. `SUNIrrep{3}("3")` and `SUNIrrep{3}("8")`.

The display of irreps can be changed in a persistent way by setting the `display_mode`
preference:

```julia-repl
julia> using SUNRepresentations
julia> for mode in ["weight", "dynkin", "dimension"]
           SUNRepresentations.display_mode(mode)
           @show SUNIrrep(2,2,2,0)
       end
SUNIrrep(2, 2, 2, 0) = Irrep[SU₄]((2, 2, 2, 0))
SUNIrrep(2, 2, 2, 0) = Irrep[SU₄]([0, 0, 2])
SUNIrrep(2, 2, 2, 0) = Irrep[SU₄]("10")
```

## Extensions

This package supports outputting the irreps to a LaTeX format via a package extension for
`Latexify.jl`. To use this extension, load `Latexify.jl` and `SUNRepresentations.jl` and
then the following should work:

```julia-repl
julia> using SUNRepresentations, Latexify
julia> latexify(SUNIrrep{4}("10⁺"))
L"$\overline{\textbf{10}}$"
```

## TODO

* Documentation
* Thread-safety of cache structures
* Ability to store/load generated cache data
