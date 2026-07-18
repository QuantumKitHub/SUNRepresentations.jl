# Explicit construction of the reduced generators (Phase 1)

How to build the SU(N) coset ladder operators as SU(N-1)×U(1) `TensorMap`s **block-by-block from
closed-form reduced matrix elements**, instead of forming a dense generator array and letting
`TensorMap(array, Vad ⊗ Va ← Va; tol)` SVD-project it into blocks (what `bootstrap.jl:57` does for
SU(3)). All references are to Molev (`molev2006`, `pdfs/arXiv-math_0211289.pdf`); page numbers are
the printed ones.

## What actually needs building

`reduced_CGC` (`bootstrap.jl:133`) only consumes **`reduced_raising_operators`** and
**`reduced_lowering_operators`** — the *coset* pieces of the adjoint (the fundamental / antifundamental
of SU(N-1) carrying U(1) charge ∓N). The other adjoint pieces are free:
- **adj(SU(N-1))** block — acts *within* each reduced sector as the SU(N-1) generators of that
  sub-irrep; supplied automatically by TensorKit's SU(N-1) sector structure. Not needed explicitly.
- **U(1) singlet** — `y · id` on each sector `(ν, y)`. Trivial.

So Phase 1 reduces to: **construct the coset raising operator** (and its adjoint, the lowering
operator) as a `TensorMap` `Va → Vfund ⊗ Va` over `SUNIrrep{N-1} ⊠ U1Irrep` sectors.

## Two structural facts that make it explicit

**(1) Degeneracy 1.** The branching gl(n)↓gl(n-1) is **multiplicity-free** (Molev Thm 2.1, p.8): each
gl(n-1) weight `μ` satisfying betweenness (Eq. 2.2) occurs exactly once in `L(λ)`. Because a reduced
charge `(ν, y)` ↔ the gl(n-1) weight `μ` bijectively (`ν` = `μ` normalized, `y` fixes the trace/row
sum), **every sector in `reduced_space(a)` has degeneracy 1**. Hence every coset-operator block is a
single scalar — the isoscalar factor for that transition. (Sanity check: for the SU(3) adjoint,
`8 → 3_{Y=0} ⊕ 1_{Y=0} ⊕ 2_{Y=3} ⊕ 2_{Y=-3}`, each `(spin, Y)` once.)

**(2) A TensorMap *subblock* = the reduced matrix element.** TensorKit stores a symmetric map as one
matrix per coupled charge (`block(t,c)`), but that matrix aggregates *all* splitting/fusion-tree
pairs reaching `c`. The reduced coefficient for one specific transition is a **subblock**, accessed
by fusion-tree indexing `t[f₁, f₂]` (`TensorKit/src/tensors/tensor.jl:457` — a view into the slice
of `block(t,c)` for the tree pair `(f₁,f₂)`). CGCs are implicit in the trees (`tensorkit2025`
Eqs. 94, 133). So we never form CGCs, dense arrays, or index into `block` by hand: we set each
subblock to its scalar reduced matrix element. The fund ⊗ ν' → ν coupling is carried by `f₁`.

## The closed-form reduced matrix elements (Molev z-operators)

Molev introduces raising/lowering operators `z_{in}`, `z_{ni} ∈ U(gl_n)` (Eqs. 2.9–2.10, p.10) that
act *between branching sectors*: for a gl(n-1)-highest-weight vector `η ∈ L(λ)⁺_μ`,

- `z_{in} η ∈ L(λ)⁺_{μ+δ_i}` (raising: adds a box in row `i` of the gl(n-1) label), and
- `z_{ni} η ∈ L(λ)⁺_{μ-δ_i}` (lowering),

(Lemma 2.5, p.11). Since each `L(λ)⁺_μ` is ≤1-dimensional (fact 1), `z_{in}` is a **scalar map**
between one-dimensional spaces, given in closed form by **Lemma 2.13 (Eq. 2.28, p.17):**

```
z_{in} ξ_μ = −(m_i − l_1)(m_i − l_2)···(m_i − l_n) · ξ_{μ+δ_i},
     with  m_i = μ_i − i + 1,   l_j = λ_j − j + 1   (j = 1..n),
```
and `ξ_{μ+δ_i} = 0` if `μ+δ_i` violates betweenness. Here `ξ_μ = z_{n1}^{λ_1−μ_1}···z_{n,n-1}^{λ_{n-1}−μ_{n-1}} ξ`
is the (unnormalized) gl(n-1)-HW vector of sector `μ` (Thm 2.7, Eq. 2.11 / Lemma 2.6, p.11).

These `ξ_μ` are **not** normalized; the orthonormal vectors are `ζ_μ = ξ_μ/‖ξ_μ‖` with the norm
from **Prop 2.4 (p.10)**. So the reduced matrix element in the orthonormal basis is
```
⟨ζ_{μ+δ_i} | z_{in} | ζ_μ⟩ = −(m_i − l_1)···(m_i − l_n) · ‖ξ_{μ+δ_i}‖ / ‖ξ_μ‖.
```

`z_{in}` is the *projected* physical generator (`z_{in} = p E_{in}·(h-factors)`, Eq. 2.17, p.15;
`p` = extremal projector), so it captures exactly the HW→HW (reduced) part of the coset generator
`E_{in}`. What remains is a single **overall normalization constant** for the tensor operator (the
`(h-factors)` and the tensor-operator norm), which we fix once (below).

## Assembly recipe (no dense array)

For irrep `a = SUNIrrep{N}(λ)`:
1. Enumerate reduced sectors `(ν, y)` of `reduced_space(a)` (existing `reduced_charge`/`reduced_space`).
2. For each sector and each **addable box** `i` (i.e. `μ+δ_i` still satisfies betweenness w.r.t. `λ`),
   the coset raising operator has a nonzero block `(ν, y) → (ν', y')`, `ν' = ν+□_i`. Its scalar
   reduced matrix element `r_{μ→μ+δ_i}` = the orthonormal `z_{in}` value above (up to the overall
   normalization).
3. Allocate `g` over `Vfund ⊗ Va ← Va` (`Vfund = SUNIrrep{N-1}` fundamental ⊠ `U1Irrep(∓N)`; note
   codomain = `Vfund ⊗ Va`, domain = `Va`, matching `bootstrap.jl:70`). Then iterate
   `for (f₁, f₂) in fusiontrees(g)` and set the **subblock** `g[f₁, f₂] = r`, where `r` is the
   scalar reduced matrix element for the transition encoded by that tree pair: `f₂` carries the
   input sector `μ_in = (ν, y)` (its coupled charge), and `f₁` couples `fund ⊗ μ_out → μ_in` with
   `μ_out = (ν', y')` the output sector. Since every sector has degeneracy 1 (fact 1), each subblock
   view is size `(1,1,1)` — assign the scalar. No dense array, no `tol`-projection, no manual `block`
   offsets.
4. `reduced_lowering_operators` = adjoint of `reduced_raising_operators` (`E_{ni} = (E_{in})^†`,
   Molev inner product, p.10) — build with `'` rather than re-deriving.

## Fixing the overall normalization & verifying (do not skip)

The `z_{in}` route gives `r` up to one scalar per fundamental tensor operator. Pin it with either:
- **Algebra gate:** reconstruct the full generator set (coset + SU(N-1) block + U(1)) and require the
  **su(N) commutation relations** and the correct quadratic **Casimir** eigenvalue. This fixes the
  normalization and the sign (resolving the `bootstrap.jl:53` vs `:78` Jm-sign inconsistency).
- **Reference-state cross-check (independent, recommended):** the SU(N-1)-highest component of the
  coset operator is `E_{N-1,N}` = `creation(a)`'s last simple operator, whose matrix elements are
  Molev Thm 2.3 Eq. 2.6 (k=N−1). Act with it on the HW GT-pattern of `ν` and read the overlap with
  the HW pattern of `ν'`, divide by the SU(N-1) HW-coupling CGC (from the recursive SU(N-1)
  `reduced_CGC` / `fusiontensor`). This must equal `r`. It uses only existing `creation` +
  lower-N CGCs and is manifestly the **bootstrap** step (level N from level N-1).

## Why this is better than the dense-projection constructor

- **No SVD/tol heuristic.** `TensorMap(array; tol)` infers blocks numerically from a dense
  `dim(Vad)·dim(Va) × dim(Va)` array; block assignment writes exact rationals/√rationals into 1×1
  blocks. Cheaper and exact.
- **Manifestly correct sectors.** Transitions are enumerated from betweenness/addable-boxes, so the
  block structure is right by construction rather than discovered up to `tol`.
- **Recursive / bootstrap.** The reference-state cross-check consumes SU(N-1) CGCs — the level-(N-1)
  result feeds level N, which is the project's core design.

## Validated prototype (N = 3, 4, 5)

Both coset operators build explicitly via per-subblock assignment and (a) reproduce the existing
SU(3) `reduced_raising_operators` / `reduced_lowering_operators` to machine precision, and (b) give
valid tensor operators for N = 3,4,5 — the per-subblock **residual**
`‖convert(Array,g) − A_raw‖/‖A_raw‖ ≈ 1e-16`. `reduced_space`/`reduced_charge` generalize to N ≥ 4
unchanged (they already emit `SUNIrrep{N-1} ⊠ U1Irrep` sectors; N = 3 keeps `SU2Irrep`).

**Discovered conventions** (match TensorKit's sector basis; found via a residual-gated
perm/sign/fund-vs-antifund search, then confirmed by the closed rule):

- **Raising** — codomain sector `antifund(SU(N-1)) ⊠ U1(−N)`. Raw components `E_{i,N}` (i = 1..N−1)
  by nested commutators `E_{N-1,N}=creation[N-1]`, `E_{i,N}=[E_{i,i+1}, E_{i+1,N}]`; stacked in
  **natural order** i = 1..N−1 with signs **(−1)^i**.
- **Lowering** — codomain sector `fund(SU(N-1)) ⊠ U1(+N)`. Raw components `E_{N,i}` by
  `E_{N,N-1}=annihilation[N-1]`, `E_{N,i}=[E_{N,i+1}, E_{i+1,i}]`; stacked in **reversed order**
  i = N−1..1 with **all signs −1**.

(For N = 3 the SU(2) factor is `SU2Irrep`, so fund = antifund; the SU(3) code's `(-E13,E23)` and
`(-E32,-E31)` are the i=1..2 / reversed special cases.) The raising operator sits in the
**antifundamental** — invisible at N = 3 (spin-½ self-dual), exposed by the N = 4 residual gate.

**Extraction (per subblock, no global tol-SVD):**
```julia
g = zeros(Float64, Vad ⊗ Va ← Va)               # Vad = coset sector above
for (f₁, f₂) in fusiontrees(g)
    gp = zero(g); (gp[f₁, f₂]) .= 1.0            # unit subblock → CGC-weighted image A_probe
    Ap = convert(Array, gp)
    (g[f₁, f₂]) .= sum(Ap .* A_raw) / sum(abs2, Ap)   # scalar r for this transition
end
```
`A_raw` = the stacked raw coset components (in the reduced basis via `reduced_basistransform`).
Runnable prototypes: `scratchpad/coset_ops.jl` (constructors + N=3,4,5 verification),
`scratchpad/prototype.jl` (SU(3) regression), `scratchpad/{gen4,lower_search}.jl` (convention search).

## Status
- **Integrated** into `src/bootstrap.jl`: `reduced_raising_operators`/`reduced_lowering_operators`
  are now general `SUNIrrep{N}` methods (helpers `_coset_fund`/`_coset_antifund`,
  `_raw_coset_raising`/`_raw_coset_lowering`, `_coset_operator`). SU(3) output is byte-for-byte the
  old result, so `reduced_CGC` and the F/R path are unchanged (smoke-tested).
- **Tests** in `test/reduced.jl` (wired into `runtests.jl`): multiplicity-free branching, scalar
  subblocks, SU(3) regression vs a frozen copy of the original implementation, representation of the
  raw SU(N) generators (the closed-form guard), and an absolute golden value (−√2). All green N=3,4,5.

## Closed form (implemented — no raw generators)

`reduced_raising_operators`/`reduced_lowering_operators` now fill each subblock from a closed-form
reduced matrix element; no dense generator is built. With `μ` the gl(N-1) weight of a reduced
sector (reconstructed from `(ν, Y)` and `λ` by `_reduced_weight`), and shifted coordinates
`m_k = μ_k − k + 1`, `l_j = λ_j − j + 1`:

- **Raising** (add box `i`, `μ_out = μ_in + δ_i`):
  `r = (−1)^i · √| ∏_{j=1}^{N}(m_i − l_j) / ∏_{k≠i}^{N−1}(m_i − m_k) |`, evaluated at `μ = μ_in`.
  (Numerator = Molev Lemma 2.13 / Eq. 2.28 product over the N level-N shifted weights; denominator =
  the level-(N−1) same-row differences, which absorbs the Prop 2.4 norm ratio + the `z_{in}`
  h-factors.)
- **Lowering** (remove box `i`, `μ_out = μ_in − δ_i`):
  `r = −√( [raising magnitude² at μ_out] · dim(ν_out)/dim(ν_in) )`. It is the Hermitian conjugate of
  the raising element `μ_out → μ_in`, rescaled by the SU(N-1) dimension ratio relating the fund and
  antifund CGC normalizations.

Verified against the raw GT generators to machine precision for N=3,4,5 (`test/reduced.jl`), and
faster than the projection (no per-subblock dense work). Committed: `301ce3a` (general operators +
tests), `18df0b3` (closed form).

## Remaining / optional
- **Algebra gate (belt-and-suspenders):** reconstruct the full generator set and check su(N)
  commutators / Casimir — not required (the raw-generator representation test already pins it).
- `reduced_generators(::SUNIrrep{3})` (the full 8-generator object, unused by `reduced_CGC`) is left
  as-is; its `:53` Jm sign is superseded by the lowering rule here if ever generalized.
- Next up: Phase 2 — `reduced_CGC` for general N (now that the coset operators exist for all N).
