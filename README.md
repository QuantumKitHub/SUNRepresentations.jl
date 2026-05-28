# SUNRepresentations.jl

[![Tests](https://github.com/QuantumKitHub/SUNRepresentations.jl/actions/workflows/Tests.yml/badge.svg)](https://github.com/QuantumKitHub/SUNRepresentations.jl/actions/workflows/Tests.yml)
[![Coverage](https://codecov.io/gh/QuantumKitHub/SUNRepresentations.jl/graph/badge.svg?token=17UEPA3KXT)](https://codecov.io/gh/QuantumKitHub/SUNRepresentations.jl)

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

As computing the Clebsch-Gordan coefficients is a relatively expensive operation, this packages automatically caches the results of the computations.
To obtain information about the current status of the cache, one can call `SUNRepresentations.cache_info()`.

Often, it may be useful to precompute a large set of coefficients (in parallel).
These can then be stored on disk and loaded when needed, or even transferred to other machines.
This can be done using the `SUNRepresentations.precompute_disk_cache(N, a_max)` function, which will compute all Clebsch-Gordan coefficients for `s1 ⊗ s2 -> s3`, where `s1` and `s2` will have Dynkin labels smaller than `a_max`, and `s3` runs over all outputs of the fusion product.

```julia-repl
julia> SUNRepresentations.precompute_disk_cache(3)
CGC disk cache info:
====================
* SU(3) - Float64 - 32 entries - 134.462 KiB
```

The values are stored at `SUNRepresentations.CGC_CACHE_PATH`, which is a package-wide
scratchspace. Each file `CGC/N/T/s1/s2.jld2` contains coefficients with datatype `T` for
the fusion of the `SU(N)` irreps `s1 ⊗ s2 → s3`, where `s3` runs over all possible fusion
channels. The folder structure is as follows:

```quote
CGC/
├── 3/
│   ├── Float64/
│   │   ├── (0, 0, 0)/
│   │   │   ├── (0, 0, 0).jld2
│   │   │   ├── (1, 0, 0).jld2
│   │   │   └── ...
│   │   ├── (1, 0, 0)/
│   │   │   └── ...
│   │   └── ...
│   └── Float32/
│      └── ...
├── 4/
└── ...
```

## Conventions

By default, irreps are denoted by their `N - 1` Dynkin labels, which are equivalent to consecutive differences in the number of boxes in each row of the Young tableau, and this is also how they are stored.
For example, the fundamental representation of SU(3) is denoted by `SUNIrrep{3}(1, 0)`, and the adjoint representation by `SUNIrrep{3}(1, 1)`.
Nevertheless, we also support using `N` weight labels, corresponding to the number of boxes per row in the Young tableaus.
For example, the fundamental representation of SU(3) is denoted by `SUNIrrep{3}(1, 0, 0)`, and the adjoint representation by `SUNIrrep{3}(2, 1, 0)`.
Finally, it is also possible to use the dimensional name which is often used in physics, e.g. `SUNIrrep{3}("3")` and `SUNIrrep{3}("8")`.

The display of irreps can be changed in a persistent way by setting the `display_mode` preference:

```julia-repl
julia> using SUNRepresentations
julia> for mode in ["weight", "dynkin", "dimension"]
           SUNRepresentations.display_mode(mode)
           @show SUNIrrep{4}(2, 2, 2, 0)
       end
SUNIrrep{4}(2, 2, 2, 0) = Irrep[SU₄]((2, 2, 2, 0))
SUNIrrep{4}(2, 2, 2, 0) = Irrep[SU₄]((0, 0, 2))
SUNIrrep{4}(2, 2, 2, 0) = Irrep[SU₄]("10")
```

## Extensions

This package supports outputting the irreps to a LaTeX format via a package extension for `Latexify.jl`.
To use this extension, load `Latexify.jl` and `SUNRepresentations.jl` and then the following should work:

```julia-repl
julia> using SUNRepresentations, Latexify
julia> latexify(SUNIrrep{4}("10⁺"))
L"$\overline{\textbf{10}}$"
```

## Breaking changes

### v0.4

v0.4 refactors the internal representation of `SUNIrrep` objects.
The change is **minimally breaking**: all public constructors and accessors continue to work as before, but any **serialized or stored** `SUNIrrep` values are incompatible with the new version.

**What changed internally:**

- `SUNIrrep{N}` previously stored an `NTuple{N, Int}` of highest-weight components (the N weights). It now stores an `NTuple{N-1, UInt8}` of Dynkin labels.
- The type now has a second type parameter: `SUNIrrep{N, M}` where `M = N - 1`.
  Code using `SUNIrrep{N}` as a type annotation continues to work; only code matching on the concrete type (e.g. `SUNIrrep{3, 2}`) or directly accessing the internal `.I` field will need updating.
- Constructors using `SUNIrrep(arg)` with `arg::Tuple` vs `arg::Vector` to distinguish between weights or dynkin labels are no longer supported, and the `N` in `SUNIrrep{N}` is now always required to avoid ambiguities.
- Direct field access `s.I` still works, but should be avoided.
  Use the public accessors `weight(s)` and `dynkin_label(s)` instead.

**Impact on stored data:**

Any data persisted to disk that contains `SUNIrrep` values will be unreadable after upgrading to v0.4.
This includes user-written **JLD2** (or similar) files, but notably **does not include** CGC disk caches.

## TODO

* Documentation
