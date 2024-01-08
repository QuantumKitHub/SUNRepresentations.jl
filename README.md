# SUNRepresentations.jl

[![Build
Status](https://github.com/maartenvd/SUNRepresentations.jl/workflows/CI/badge.svg)](https://github.com/maartenvd/SUNRepresentations.jl/actions)
[![Coverage](https://codecov.io/gh/maartenvd/SUNRepresentations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/maartenvd/SUNRepresentations.jl/branch/master)

Compute Clebsch-Gordan coefficients for general SU(N) groups. Reimplementation of [arXiv:1009.0437](https://arxiv.org/pdf/1009.0437.pdf). Compatibility / interoperability with [TensorKit.jl](https://github.com/Jutho/TensorKit.jl).

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
* Ability to store/load generated cache data
