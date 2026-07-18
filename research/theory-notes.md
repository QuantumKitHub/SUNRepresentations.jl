# Theory notes: SU(N-1)×U(1)-symmetric CGCs as TensorMaps

These notes connect the representation theory to the code on the `reduced` branch
(`src/bootstrap.jl`, `src/gtpatterns.jl`, `src/sector.jl`). They are working notes, not a paper —
sign/normalization conventions follow the code and should be re-derived when in doubt.

## 1. The subgroup chain and Gelfand-Tsetlin patterns

An SU(N) irrep is labelled by a highest weight (Young-tableau row lengths) `SUNIrrep{N}`. Its basis
states are **Gelfand-Tsetlin (GT) patterns** (`GTPattern{N}`), triangular arrays whose rows encode
a chain of restrictions

```
SU(N) ⊃ SU(N-1) ⊃ … ⊃ SU(2) ⊃ SU(1).
```

Row `N-1` of a GT pattern is precisely the SU(N-1) irrep the state belongs to under
SU(N) ↓ SU(N-1). This is the structural fact the whole approach rests on: **the GT basis is already
adapted to the subgroup chain.**

### Reduced charge (`reduced_charge`, `src/bootstrap.jl:8`)
Each state carries a well-defined SU(N-1)×U(1) charge:
- **SU(N-1) part:** the irrep read off row `N-1`, normalized so the last entry is 0
  (`_normalize`). (For N=3 the code currently reinterprets the SU(2) label as `SU2Irrep(I/2)`; the
  general-N path uses `SUNIrrep{N-1}`.)
- **U(1) hypercharge:** `Y = N·rowsum(g, N-1) − (N-1)·rowsum(g, N)`, chosen so the total charge
  over a multiplet is traceless (the SU(N)-consistent normalization).

`reduced_space` (`bootstrap.jl:17`) bins the basis by reduced charge to build a graded
`Vect[SUNIrrep{N-1} ⊠ U1Irrep]`; `reduced_basistransform` (`bootstrap.jl:28`) is the permutation
that reorders the raw GT basis into these blocks so the ladder-operator matrices become
block-structured.

## 2. Racah factorization / isoscalar factors — why a CGC *is* a TensorMap

The **Racah factorization lemma** states that a CGC of `G = SU(N)` factorizes, along the chain
`G ⊃ H = SU(N-1)×U(1)`, as

```
⟨ a, m_a ; b, m_b | c, m_c ⟩_G  =  ⟨ α_a ; α_b | α_c ⟩_H  ·  [ a b ; c ]^{H}_{...}
```

i.e. an `H`-CGC (depending on the `H`-internal labels `α`) times an **isoscalar factor** (reduced
Wigner coefficient) that is *independent* of those internal labels. Iterating down the chain
expresses any SU(N) CGC as a product of isoscalar factors — one per link SU(k)⊃SU(k-1)×U(1).

In TensorKit language, an `H`-covariant map whose blocks are the isoscalar factors and whose
`H`-CGC part is supplied automatically by the fusion machinery is exactly a **`TensorMap` over
`H`-graded spaces**. So:

> The SU(N) CGC `a ⊗ b → c` is stored as `reduced_CGC(a,b,c) :: TensorMap` over
> `reduced_space(a) ⊗ reduced_space(b) ← reduced_space(c) ⊗ Vn`,
> where `Vn` is the outer-multiplicity space (`bootstrap.jl:137`).

This is exact, not approximate: the isoscalar factors are the reduced (nonzero) blocks, and the
storage/compute savings come from only ever touching those blocks — the same win non-abelian
symmetry gives tensor networks (QSpace, `weichselbaum2012`).

## 3. Generators as a rank-1 tensor operator over the adjoint

Under SU(N)↓SU(N-1)×U(1) the **adjoint** representation decomposes as

```
adj(SU(N))  ↓  adj(SU(N-1))  ⊕  1  ⊕  fund(SU(N-1))  ⊕  antifund(SU(N-1)),
dim:  N²-1   =   (N-1)²-1     +  1  +     (N-1)        +      (N-1).
```

- **adj(SU(N-1))** — the SU(N-1) subalgebra generators.
- **1** — the U(1) (hypercharge) generator.
- **fund**, charge `−N` — the **raising** operators that step *up* in U(1) charge.
- **antifund**, charge `+N` — the **lowering** operators.

So the full set of generators is a single **rank-1 tensor operator** transforming in
`reduced_space(adjoint_irrep(a))`. In code this is a `TensorMap` over
`reduced_space(adj) ⊗ reduced_space(a) ← reduced_space(a)` (`reduced_generators`, `bootstrap.jl:38`,
currently SU(3)-only). For SU(3): `8 → 3 ⊕ 1 ⊕ 2 ⊕ 2̄`, matching the manual `stack` of
`adjoint_generators`/`singlet_generator`/`raising`/`lowering` blocks.

The Wigner-Eckart theorem then says: knowing how the raising/lowering blocks act between reduced
sectors (plus the fixed `H`-CGCs) fully determines the operator — which is why we can build the
CGC purely from the reduced ladder operators.

## 4. The CGC equations, in reduced form

`reduced_CGC` (`bootstrap.jl:133`) is the reduced-basis mirror of the dense two-stage algorithm
(`alex2011`; `src/clebschgordan.jl`):

**Highest weight.** The highest-weight coupling is annihilated by the combined raising operators,
`(J⁺_a ⊗ 1 + 1 ⊗ J⁺_b) · CG[hw] = 0`. Restricted (via `highest_weight_projector`) to the fused
highest-weight sector, the solution is the **right null space** of that operator
(`rightnull(HW_eq; alg=SDD())`, `bootstrap.jl:148`). Its dimension is the outer multiplicity
`Nsymbol(a,b,c)`; a `gaugefix!` picks a canonical basis among degenerate multiplicities.

**Lower weights.** Apply lowering operators and solve the recursion
`CG[w-1]·J⁻_c = (J⁻_a ⊗ 1 + 1 ⊗ J⁻_b)·CG[w]` sector by sector (`bootstrap.jl:159` loop), stepping
the weight projector down with `Jmc`. `heightmap` (`bootstrap.jl:108`) encodes the lowering DAG and
could drive a provably-complete descent.

## 5. The bootstrap recursion (the point of the whole thing)

To build `reduced_generators(SU(N))` you need to know how the SU(N-1) subalgebra acts in *its* own
reduced (SU(N-2)×U(1)) basis — i.e. `reduced_generators(SU(N-1))`. Hence the construction is
recursive:

```
SU(2)  →  SU(3)  →  SU(4)  →  …  →  SU(N)
(base)     built from SU(2), etc.
```

Each level reuses the ladder operators of the level below (available for any N from
`creation`/`annihilation` in `src/gtpatterns.jl`), reorganized into the SU(N-1)×U(1) block
structure. Getting the adjoint-index ordering and normalization consistent with TensorKit's sector
ordering is the main engineering task (roadmap Phase 1).

## 6. F- and R-symbols

Once CGCs are correct reduced `TensorMap`s, the topological data follows by contracting them
(`_Fsymbol`/`_Rsymbol`, `src/sector.jl`). The current `/dim(d)`, `/dim(c)` factors are placeholders
(`# TODO: replace trace with projector`); the correct, multiplicity-aware contraction is the
**X-symbol** construction of `weichselbaum2020`. Correctness is pinned by the pentagon, hexagon,
and F-unitarity checks already in `test/sectors.jl`.

## References
See `references.md` / `references.bib`. Core: `alex2011`, `weichselbaum2012`, `weichselbaum2020`,
`tensorkit2025`. The isoscalar-factor / Wigner-Racah strand (§6) is the classical version of this
construction: GT bases (`gelfand1950`, `molev2006`), U(n) pattern calculus / factorization lemma
(`biedenharn1968`, `louck1970`, `biedenharn1981`), SU(3) isoscalar factors and coefficients
(`deswart1963`, `hecht1965`, `draayer1973`, `kaeding1995` for validation numbers), the VCS
induction of SU(3) coefficients from SU(2)×U(1) — closest methodological analog — (`leblanc1990`),
and the outer-multiplicity subtlety for SU(N) (`williams1996`, `pan1998`).
