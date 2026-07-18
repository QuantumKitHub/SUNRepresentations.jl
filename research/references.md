# Annotated references

Literature relevant to computing SU(N) Clebsch-Gordan coefficients (CGCs) and to the
SU(N-1)×U(1)-symmetric / TensorMap approach explored on the `reduced` branch.

BibTeX keys refer to `references.bib`.

---

## 1. The current algorithm

### `alex2011` — Alex, Kalus, Huckleberry, von Delft (2011)
*A numerical algorithm for the explicit calculation of SU(N) and SL(N,C) Clebsch-Gordan
coefficients*, J. Math. Phys. **52**, 023507. arXiv:[1009.0437](https://arxiv.org/abs/1009.0437).

The reference the current `src/clebschgordan.jl` reimplements. Works for arbitrary N and arbitrary
pairs of irreps. Uses **Gelfand-Tsetlin (GT) pattern calculus**: each irrep is a highest weight,
basis states are GT patterns (triangular arrays obeying betweenness conditions), and the ladder
(raising/lowering) operators have closed-form matrix elements. The CGC computation proceeds in two
stages that this package mirrors exactly: (i) the **highest-weight** CGC is the null space of the
combined raising operators `(J⁺₁⊗1 + 1⊗J⁺₂)`, whose dimension equals the outer multiplicity;
(ii) **lower-weight** coefficients follow by repeatedly applying lowering operators. A companion
online generator and code are provided by the authors. *Relevance:* the ground-truth algorithm and
the dense reference our reduced results must reproduce.

---

## 2. Non-abelian symmetry as a tensor-network structure (the conceptual parent)

### `weichselbaum2012` — Weichselbaum (2012)
*Non-abelian symmetries in tensor networks: a quantum symmetry space approach*, Ann. Phys.
**327**, 2972. arXiv:[1202.5664](https://arxiv.org/abs/1202.5664).

The **QSpace** framework. Shows that, in the presence of a non-abelian symmetry, every symmetric
tensor factorizes (generalized Wigner-Eckart theorem) into a **reduced-matrix-element tensor**
(the physical data) times a **generalized Clebsch-Gordan tensor** (CGT, fixed by representation
theory). Crucially, it computes CGCs on the **Lie-algebra level** and organizes multiplets via the
**subgroup chain** SU(2)⊂SU(3)⊂…⊂SU(N). This is the direct conceptual ancestor of the `reduced`
branch: "a CGC is an SU(N-1)×U(1)-covariant object" is exactly the QSpace viewpoint, and TensorKit's
`TensorMap` plays the role of the CGT. *Relevance:* justifies the whole approach and gives the
bootstrap-over-subgroup-chain recipe.

---

## 3. Recoupling / F- and R-symbols from CG tensors

### `weichselbaum2020` — Weichselbaum (2020)
*X-symbols for non-Abelian symmetries in tensor networks*, Phys. Rev. Research **2**, 023385.
arXiv:[1910.13736](https://arxiv.org/abs/1910.13736).

Introduces **X-symbols**: a systematic, bottom-up way to handle pairwise contractions of
generalized CG tensors for general non-abelian symmetries (SU(N), Sp(2n), SO(n)). X-symbols are as
general as 3n-j symbols — any 3n-j symbol (hence any F-/R-type recoupling coefficient) can be
computed from them. *Relevance:* the recipe for Phase 3 — deriving correctly-normalized
`Fsymbol`/`Rsymbol` from the TensorMap CGCs, replacing the current `/dim(d)` placeholder in
`src/sector.jl`.

---

## 4. The TensorMap / fusion-category framework

### `tensorkit2025` — Van Damme, Devos, Haegeman et al. (2025)
*TensorKit.jl: A Julia package for large-scale tensor computations, with a hint of category
theory*. arXiv:[2508.10076](https://arxiv.org/abs/2508.10076).
Docs: https://jutho.github.io/TensorKit.jl/ (sectors, graded spaces, fusion trees).

Defines the `TensorMap`, `GradedSpace`/`Vect[I]`, and the topological data (`Nsymbol`, `Fsymbol`,
`Rsymbol`, `fusiontensor`) that `SUNRepresentations.jl` must supply. A `GradedSpace` grades a vector
space by the simple objects (irreps) of a fusion category; a basis change from a tensor product of
irreps into a fused irrep *is* a sequence of CGCs (a fusion/splitting tree). *Relevance:* the target
interface — the reduced CGCs are `TensorMap`s over SU(N-1)×U(1) `GradedSpace`s, and the sector
interface in `src/sector.jl` is exactly TensorKit's expected API.

---

## 5. Racah factorization & isoscalar factors (the mathematical justification)

The statement "an SU(N) CGC decomposes through the chain SU(N)⊃SU(N-1)×U(1)" is the **Racah
factorization lemma**: a coupling coefficient of `G` factorizes into a coupling coefficient of the
subgroup `H` times a **reduced coupling coefficient** (isoscalar factor / reduced Wigner
coefficient) that is independent of the `H`-internal labels. Iterating down the chain expresses any
SU(N) CGC as a product of isoscalar factors. This is why the CGC is fully captured by the reduced
blocks of a `TensorMap`, and it underlies the SU(3)→SU(4) recursion.

Representative sources (journal-gated; link a free preprint where one exists):
- `su4cgc` — *Program for calculating SU(4) Clebsch–Gordan coefficients*, Comput. Phys. Commun.
  ([S0010465508002300](https://www.sciencedirect.com/science/article/abs/pii/S0010465508002300)):
  SU(4) CGCs built from isoscalar factors and SU(3) CGCs via Racah factorization — a concrete
  instance of the bootstrap.
- Racah's method for general subalgebra chains (coupling coefficients via reduced coefficients);
  and the broad Wigner–Racah / isoscalar-factor literature for SU(3) (e.g. recursion relations for
  SU(3) isoscalar factors).

*Relevance:* the theorem that makes the TensorMap representation exact rather than approximate;
see `theory-notes.md` for how it maps onto the code.

---

## 6. Mathematical & SU(3)/SU(N) isoscalar-factor literature

This project is, in essence, **isoscalar-factor computation for SU(N)** phrased in TensorKit
language: `reduced_CGC` blocks *are* SU(N)⊃SU(N-1)×U(1) isoscalar factors, and the bootstrap over
the subgroup chain is exactly the classical strategy of inducing SU(n) coupling coefficients from
SU(n-1). The following is the core math-physics literature; the SU(3) case is the most developed
and the closest methodological analog.

### 6a. Foundations: U(n)/SU(n) Wigner-Racah algebra & the GT basis
- `gelfand1950` — **Gelfand & Tsetlin (1950)**, *Finite-dimensional representations of the group of
  unimodular matrices*, Dokl. Akad. Nauk SSSR **71**, 825. Origin of the GT basis / patterns
  (`GTPattern`) and the subgroup-chain labelling this whole approach uses.
- `biedenharn1968` — **Biedenharn & Louck (1968)**, *A pattern calculus for tensor operators in the
  unitary groups*, Commun. Math. Phys. **8**, 89. The **factorization lemma** and pattern calculus:
  reduced matrix elements / isoscalar factors from combinatorial "arrow patterns". The theoretical
  backbone of "generators as a rank-1 tensor operator over the adjoint".
- `louck1970` — **Louck (1970)**, *Recent progress toward a theory of tensor operators in the
  unitary groups*, Am. J. Phys. **38**, 3. Readable synthesis of the U(n) Racah-Wigner program.
- `biedenharn1981` — **Biedenharn & Louck (1981)**, *Angular Momentum in Quantum Physics* (Encycl.
  Math. Appl. 8) & *The Racah-Wigner Algebra in Quantum Theory* (vol. 9), Addison-Wesley. The
  standard monographs.
- `molev2006` — **Molev (2006)**, *Gelfand-Tsetlin bases for classical Lie algebras*, Handbook of
  Algebra **4**; arXiv:[math/0211289](https://arxiv.org/abs/math/0211289). Modern, rigorous
  treatment of GT bases and matrix-element formulae — a clean reference for the ladder-operator
  matrix elements in `src/gtpatterns.jl`. **PDF in `pdfs/`.**

### 6b. SU(3) isoscalar factors / Wigner-Racah — the canonical strand
- `deswart1963` — **de Swart (1963)**, *The octet model and its Clebsch-Gordan coefficients*, Rev.
  Mod. Phys. **35**, 916 (erratum 1965). Introduced SU(3) isoscalar factors to physics; classic
  reference tables. (Journal-gated.)
- `hecht1965` — **Hecht (1965)**, *SU₃ recoupling and fractional parentage in the 2s-1d shell*,
  Nucl. Phys. **62**, 1. Explicit algebraic SU(3) Wigner **and** Racah coefficients in the
  SU(3)⊃SU(2) chain. (Journal-gated.)
- `draayer1973` — **Draayer & Akiyama (1973)**, *Wigner and Racah coefficients for SU₃*, J. Math.
  Phys. **14**, 1904; plus `akiyama1973` — **Akiyama & Draayer (1973)**, *A user's guide to Fortran
  programs for Wigner and Racah coefficients of SU₃*, Comput. Phys. Commun. **5**, 405. The
  workhorse algorithm/code for decades; note the SU(3)⊃SU(2)×U(1) construction and how Racah
  coefficients drop out as a by-product. (Journal-gated.)
- `leblanc1990` — **Le Blanc & Rowe (1990)**, *Vector coherent state constructions of U(3)
  symmetric tensors and their SU(3)⊃SU(2)×U(1) Wigner coefficients*, J. Math. Phys. **31**, 2781.
  **Most direct methodological analog:** VCS theory *induces* SU(3) coupling coefficients from those
  of the SU(2)×U(1) subgroup — precisely the bootstrap this project does, in an operator language.
  (Author order to verify against the published version; journal-gated.)
- `kaeding1995` — **Kaeding (1995)**, *Tables of SU(3) isoscalar factors*, At. Data Nucl. Data
  Tables **61**, 233; arXiv:[nucl-th/9502037](https://arxiv.org/abs/nucl-th/9502037). Extensive
  tabulated isoscalar factors — **ready-made reference values for validating `reduced_CGC` at N=3.**
  **PDF in `pdfs/`.**
- `su3lib2021` — **SU3lib (2021)**, *A C++ library for accurate computation of Wigner and Racah
  coefficients of SU(3)*, Comput. Phys. Commun. (Langr, Dytrych, et al.). Modern, numerically
  robust reimplementation of the Draayer-Akiyama algorithm — a performance/accuracy benchmark.
  (Author list/details to verify; journal-gated.)

### 6c. General SU(N) via canonical factorization & the outer-multiplicity problem
- `williams1996` — **Williams (1996)**, *SU3 isoscalar factors*, J. Math. Phys. **37**, 4187;
  arXiv:[hep-th/9509167](https://arxiv.org/abs/hep-th/9509167). Derives **recursion relations** for
  SU(3) isoscalar factors and an algorithm to solve them — the recursive spirit of the bootstrap.
  **PDF in `pdfs/`.**
- `pan1998` — **Pan & Draayer (1998)**, *A complementary group technique for the resolution of the
  outer multiplicity problem of SU(n)*, J. Math. Phys. **39**, 5642;
  arXiv:[quant-ph/9704015](https://arxiv.org/abs/quant-ph/9704015) (part II; see also part I).
  **Directly relevant to the multiplicity space `Vn`:** the canonical SU(n)⊃U(1)×SU(n-1)
  factorization resolves *weight* multiplicity but not the *outer* (representation) multiplicity in
  the CG series — the subtlety flagged in roadmap Phase 2. **PDF in `pdfs/`.**
- `su4cgc` — *Program for calculating SU(4) Clebsch–Gordan coefficients* (see §5): the concrete
  SU(4)-from-SU(3) instance of the bootstrap.

**Takeaway for this project.** The "novel" TensorMap approach is a modern re-encoding of a
well-trodden idea (Racah factorization / VCS induction along SU(n)⊃SU(n-1)×U(1)). Two payoffs from
this literature: (i) `kaeding1995` gives independent reference numbers to validate N=3; (ii)
`pan1998`/`williams1996` warn that outer multiplicity needs explicit care beyond the plain
subgroup factorization — informing how `Vn` and `gaugefix!` must be handled at N>3.
