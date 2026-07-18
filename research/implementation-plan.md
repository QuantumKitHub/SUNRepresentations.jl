# Implementation plan — SU(N-1)×U(1) TensorMap CGCs

Synthesis of the codebase analysis and the literature in this folder into an actionable, phased
plan. Citations point to `references.md` keys and to `pdf-guide.md` (which has the section/page/eq
pointers). This refines the high-level roadmap in the approved plan file with concrete steps,
validation gates, and the design decisions the paper-reading surfaced.

## Goal & strategy

Represent SU(N) Clebsch-Gordan coefficients as TensorKit `TensorMap`s over SU(N-1)×U(1) sectors,
so the reduced blocks are SU(N)⊃SU(N-1)×U(1) **isoscalar factors** (Racah factorization,
`theory-notes.md`). This yields non-abelian storage/compute savings and is built up the chain
SU(2)⊂SU(3)⊂…⊂SU(N). The construction mirrors the classical isoscalar-factor program
(`leblanc1990` induces SU(3) from SU(2)×U(1); `weichselbaum2012` Fig. 13 is the operator-level
algorithm) but targets the TensorKit interface (`tensorkit2025`).

**Key realization from the reading:** the reduced generators needed by `reduced_CGC` are the
*coset* (fund/antifund) ladder operators, and their SU(N-1)×U(1) reduced matrix elements have a
**closed form** (Molev z-operators, Lemma 2.13 / `molev2006`). Because gl(n)↓gl(n-1) is
multiplicity-free, every reduced sector has degeneracy 1, so these are *scalars* — one per
addable-box transition. They can be written **directly into TensorMap blocks** (a block = the
reduced matrix element in TensorKit), so Phase 1 needs no dense-array projection and no new
representation theory beyond transcribing/verifying the closed form. See
`generator-construction.md`.

## Current state (branch `reduced`)

> **Note (2026-07-18):** the snapshot below is the *original* pre-Phase-1 state. Phases 0–3 are now
> complete — see the per-phase **STATUS** blocks. In short: general-N `reduced_CGC` (Phases 1–2) and
> F/R via a compact projector contraction on `reduced_CGC` (Phase 3) work, with the category gates
> green at N=3,4. Phases 4–5 (cache restore, `runtests.jl` wiring, README note) remain.

Works end-to-end for **SU(3) only**. `reduced_CGC` (`bootstrap.jl:133`) is generic in form but
calls SU(3)-only operators. F/R (`sector.jl`) already route through `reduced_CGC` with a placeholder
`/dim(d)`. SU(3)-only: `reduced_generators` (`:38`), `reduced_raising_operators` (`:61`),
`reduced_lowering_operators` (`:74`). Generic already: `reduced_charge` (`:8`), `reduced_space`
(`:17`), `reduced_basistransform` (`:28`), `reduced_CGC` (`:133`), `highest_weight_projector`
(`:106`), `heightmap` (`:108`, unused).

---

## Phase 0 — Groundwork & cleanup (unblock the rest)

**Goal:** a clean, uniform base so Phase 1 isn't fighting SU(3) special-casing or known bugs.

1. **Uniform reduced sector type.** `reduced_charge`/`reduced_space`/`weight_space` special-case
   `N==3 ? SU2Irrep : SUNIrrep{N-1}` (`bootstrap.jl:12,18,89`). Decide the base case: keep
   `SU2Irrep` for N=3 (best TensorKit support, keeps current SU(3) tests valid) but make the
   `SUNIrrep{N-1}` path a first-class citizen for N≥4. Confirm `SUNIrrep{2}` behaves as a sector
   (Nsymbol, dual, etc.) so the recursion has a consistent story. The stored sector is always
   `SUNIrrep{N-1} ⊠ U1Irrep` — **flat, not nested** (the SU(N-1) factor is a full irrep, so no
   nested product sectors are needed).
2. **Fix the Jm sign inconsistency:** `reduced_generators` uses `stack((Jm2, -Jm3))`
   (`bootstrap.jl:53`) but `reduced_lowering_operators` uses `stack((-Jm2, Jm3))` (`:78`). Resolve
   by deriving the sign from the tensor-operator/adjoint convention (Molev orthonormal form) and
   pin it with a commutator test (Phase 1 validation catches this).
3. **Remove dead code:** the overwritten `g = zeros(...)` (`bootstrap.jl:82`); commented
   `fusiontensor` alternates in `sector.jl`.
4. **Cache decision (defer restore to Phase 4, but note now):** `_get_CGC`
   (`clebschgordan.jl:25`) has an unconditional early `return generate_CGC(...)` making the
   RAM/disk cache dead. Leave as-is during development; Phase 4 restores it for the TensorMap path.

**Done when:** SU(3) still passes all existing tests with the SU2Irrep base case, and the codepaths
that will be generalized are free of the sign ambiguity and dead code.

---

## Phase 1 — General-N reduced generators, built explicitly (the crux)

**Goal:** `reduced_raising_operators` and `reduced_lowering_operators` for `SUNIrrep{N}`, any N,
**constructed block-by-block from closed-form reduced matrix elements** — *not* by forming a dense
generator array and SVD-projecting it with `TensorMap(array, Vad ⊗ Va ← Va; tol)` (the SU(3)
approach at `bootstrap.jl:57`). Full derivation: **`generator-construction.md`**.

**What must be built.** `reduced_CGC` only consumes the **coset** operators (the fund/antifund of
SU(N-1) carrying U(1) charge ∓N). The other adjoint pieces are free: the **adj(SU(N-1))** block acts
within each sector as the SU(N-1) generators (TensorKit's sector structure supplies it), and the
**U(1) singlet** is `y·id`. So Phase 1 = construct the coset raising operator (lowering = its
adjoint).

**Two facts that make it explicit** (`generator-construction.md`, from `molev2006`):
1. gl(n)↓gl(n-1) is **multiplicity-free** (Molev Thm 2.1) ⇒ every reduced sector has **degeneracy 1**
   ⇒ each coset block is a single scalar (an isoscalar factor).
2. In TensorKit the reduced matrix element for one transition is a **subblock**, set by fusion-tree
   indexing `g[f₁, f₂]` (not `block(g,c)`, which aggregates all tree pairs reaching `c`); CGCs live
   in the trees (`tensorkit2025` Eqs. 94, 133; API `TensorKit/src/tensors/tensor.jl:457`). So we set
   scalars into subblocks; TensorKit supplies the fund⊗ν'→ν coupling via `f₁`.

**Closed form.** Molev's raising operator `z_{in}` maps the SU(N-1)-HW vector of sector `μ` to that
of `μ+δ_i` (add box in row `i`), with value (**Lemma 2.13, Eq. 2.28**)
`z_{in} ξ_μ = −(m_i−l_1)···(m_i−l_n) ξ_{μ+δ_i}`, `m_i=μ_i−i+1`, `l_j=λ_j−j+1`. Orthonormalize with
the Prop 2.4 norm ratio `‖ξ_{μ+δ_i}‖/‖ξ_μ‖`. This is the reduced matrix element up to one overall
tensor-operator normalization.

**Steps**
1. Generalize `reduced_charge`/`reduced_space` to `SUNIrrep{N-1} ⊠ U1Irrep` (Phase 0), keeping the
   SU(3) `SU2Irrep` base case.
2. For irrep `a`, enumerate reduced sectors and their **addable boxes** (betweenness-valid `μ+δ_i`);
   each gives one coset transition `(ν,y)→(ν+□_i, y')`.
3. Compute the scalar reduced matrix element `r` for each transition from Eq. 2.28 + Prop 2.4.
4. Assemble `g` over `Vfund ⊗ Va ← Va` (`Vfund = fund(SU(N-1)) ⊠ U1Irrep(∓N)`, generalizing
   `bootstrap.jl:68,81`) by iterating `for (f₁,f₂) in fusiontrees(g)` and setting the **subblock**
   `g[f₁,f₂] = r` (degeneracy 1 ⇒ a `(1,1,1)` view ⇒ a scalar). Lowering = `'`.

**Normalization & verification (this pins signs/normalization):**
- **Reference-state cross-check (recommended, and it's the bootstrap step):** the SU(N-1)-HW
  component of the coset operator is `E_{N-1,N}` = the last `creation(a)` operator (Molev Thm 2.3
  Eq. 2.6, k=N−1). Act on the HW GT-pattern of `ν`, read the overlap with the HW pattern of `ν'`,
  divide by the SU(N-1) HW-coupling CGC (from the recursive level-(N-1) `reduced_CGC`/`fusiontensor`).
  Must equal `r`. Uses only existing code + lower-N CGCs.
- **Algebra gate:** reconstruct the full generators (coset + SU(N-1) + U(1)) and require the su(N)
  commutation relations and the correct quadratic **Casimir** eigenvalue. Fixes the overall
  normalization and the Jm sign (`bootstrap.jl:53` vs `:78`).

**Why explicit beats the dense constructor:** exact 1×1 block entries instead of an SVD/`tol`
inference over a `dim(Vad)·dim(Va)×dim(Va)` array; correct sectors by construction (addable boxes),
not discovered up to tolerance; and it is manifestly recursive (level N from level N-1).

**Done when:** both coset operators build for N=3,4,5, pass the reference-state cross-check and the
commutator + Casimir gates, and reproduce the existing SU(3) operators (up to the resolved sign).

**STATUS — DONE (2026-07-16), committed `301ce3a` + `18df0b3`.** `reduced_raising_operators` /
`reduced_lowering_operators` in `src/bootstrap.jl` build both coset operators for any N from a
**closed-form reduced matrix element** (no raw generators), verified vs the raw GT generators to
machine precision for N=3,4,5 by `test/reduced.jl`. Formulas + derivation: `generator-construction.md`
and `phase1-derivation.typ`/`.pdf`. Conventions: raising = antifund(SU(N-1))⊠U1(−N), lowering =
fund(SU(N-1))⊠U1(+N); helpers `_reduced_weight`, `_raise_mag2`, `_coset_fund`/`_coset_antifund`.
SU(3) output is unchanged, so `reduced_CGC` and F/R are unaffected.

---

## Phase 2 — `reduced_CGC` for general N  ← START HERE (next session)

**Kickoff notes for a fresh session.** Phase 1 is done and committed; the coset operators exist for
all N. Environment is ready: the local (gitignored) `Manifest.toml` is resolved with TensorKit
0.14.11, so `julia --project=.` works directly — no scratch env needed. Run the reduced tests with
`julia --project=. -e 'include("test/reduced.jl")'` or the full suite via `Pkg.test()`. Julia 1.12.6.
First concrete target: get `reduced_CGC(a,b,c)` running and correct for **N=4** (it already works for
N=3), then validate against the dense `CGC` from `clebschgordan.jl` (up to the multiplicity gauge)
and add a `test/reduced.jl` testset for it. Key reference values: Kaeding SU(3) tables for N=3.

**Goal:** `reduced_CGC(a,b,c)` correct for any N, with outer multiplicity handled.

The routine (`bootstrap.jl:133`) is already generic; once Phase 1 supplies the operators it should
run. Harden it:

1. **Highest-weight block** (`:142–151`): `HW_eq` = combined raising ops on the HW-projected fused
   space; `CG_w = P_hw · rightnull(HW_eq; alg=SDD())'` (Alex Eq. 36 / `alex2011`; QSpace Fig. 13 /
   `weichselbaum2012`). The null-space dimension = `Nsymbol(a,b,c)` = `dim(Vn)`. **Verify the
   uniqueness assumption** flagged at `:141`: handle the case where the HW fused sector is itself
   degenerate.
2. **Outer multiplicity `Vn`** (`:137`): keep it as an explicit multiplicity leg (matches QSpace's
   "orthogonal CGC copies, not an extra index" design, `weichselbaum2012` §II.C, and TensorKit's
   multiplicity leg). Gauge-fix the multiplicity basis with the existing `gaugefix!`
   (`clebschgordan.jl:50`). **Design commitment from `pan1998`:** in degenerate sectors the basis is
   an SO(m) gauge *choice* — pick a deterministic canonical basis (e.g. `gaugefix!`'s
   RREF+QR-pos, analogous to Alex Eq. 37 / lower-triangular of Pan-Draayer Eq. 2.21) and document
   it. Do not expect a gauge-independent value.
3. **Lower-weight recursion** (`:159–167`): `CG[w-1]·Jmc = (Jma⊗1 + 1⊗Jmb)·CG[w]` (Alex Eq. 40).
   Confirm the `while dim(Pw)!=0` descent visits every reduced sector; if not provably complete,
   drive it from `heightmap` (`bootstrap.jl:108`, currently unused) — the lowering DAG guarantees
   every weight is reached and parents-before-children ordering.

**Validation (correctness gate — the crucial one):**
- **Reduced-vs-dense:** `convert(Array, reduced_CGC(a,b,c))` ≈ `CGC(Float64,a,b,c)` from
  `clebschgordan.jl`, **up to the multiplicity gauge**, for a sampled set at N=3,4,5. (Fix a common
  gauge or compare the spans in degenerate sectors.)
- **N=3 against literature:** compare multiplicity-free products to Kaeding's tables
  (`kaeding1995`, `pdf-guide.md` §6a) — match conventions (SU(2) leg in Condon-Shortley; overall
  sign per their Eq. 1). For the one degenerate case (15⊗8) compare up to SO(m) (`pan1998`).
- **Isometry:** `reduced_CGC` must satisfy `C†C = 1` on `Vc⊗Vn` (TensorKit splitting-tensor
  requirement, `tensorkit2025` Eqs. 138–140).

**Done when:** the reduced-vs-dense gate passes for a representative sample at N=3,4,5 and the N=3
Kaeding spot-checks match.

**STATUS — DONE (2026-07-17), committed `76e75e7`.** `reduced_CGC(a,b,c)` is correct for N=3,4,5.
Validated *intrinsically* (isometry + coset raising/lowering intertwiners — a map that is isometric
and intertwines all su(N) generators is the CGC up to gauge; no fragile dense-array bridge needed)
plus an SU(3) isoscalar-factor cross-check against de Swart/Kaeding values. Two fixes were required
vs the "already works, just harden" assumption: (a) `_Fsymbol`/`_Rsymbol` reverted to the dense
`fusiontensor` path (the `/dim(d)` reduced form was inconsistent with densification and corrupted
`@tensor` contractions at N≥4); (b) `gaugefix!`'s return value must be assigned into the HW block
(`bootstrap.jl:188`), else degenerate HW sectors were non-orthonormal. Derivation:
`phase2-derivation.typ`/`.pdf`.

**Update (2026-07-18):** the `/dim(d)` reduced F/R path was *re-enabled and completed* in Phase 3 (see
that STATUS block). The N≥4 breakage cited in (a) was not the `/dim(d)` idea itself but three separate
issues — an outer-multiplicity gauge mismatch, a `rightnull` tolerance drop, and descent recoupling
noise — now all fixed. The `/dim(d)` trace *is* numerically correct (Schur); its only downside is a
float-accumulation residual, mitigated by the refinement and noted as a reference-state follow-up.

---

## Phase 3 — F- and R-symbols from TensorMap CGCs

**Goal:** replace the placeholder normalization in `_Fsymbol`/`_Rsymbol` (`sector.jl:86,118`) with
the correct, multiplicity-aware contraction.

**Insight from `weichselbaum2020` (`pdf-guide.md` §3):** the current `/dim(d)` is *numerically
correct* — since `reduced_CGC` is an isometry, Schur's lemma gives `Tree_R†∘Tree_L = F ⊗ 1_d`, so
tracing the `d` line yields `dim(d)·F`. But it wastefully traces the whole `d` line. The
"replace trace with projector" fix (X-symbols Eq. 7/16):

1. Contract the shared internal line (`d` for F, `c` for R) through
   `highest_weight_projector(Vd, d)` (`bootstrap.jl:106`) onto its HW sub-block instead of tracing
   the full line; normalize by that block's `blockdim` (or evaluate on one reference state, no
   division). Cheaper and avoids mixing numerical noise across `dim(d)` states.
2. **Keep all four OM legs open** (already correct) — for SU(N≥3) fusion multiplicities >1, so F is a
   genuine rank-4 tensor over OM indices (X-symbols Eq. 17); they cannot be dropped.
3. Ensure leg order/arrows match `reduced_CGC`'s `(a,b; c,μ)` ordering (isometry convention is
   order-dependent; `tensorkit2025` §4).

**Validation (category gate):** the existing `test/sectors.jl` **pentagon** and **hexagon**
equations, **F-unitarity** (`F'F ≈ I`), and fusiontensor↔F/R consistency must pass for
`SUNIrrep{3,4,5}` (`tensorkit2025` Eqs. 147, 158–159). SU(2) (no rank-3 OM) gives scalar F/R — a
clean regression case.

**Done when:** F/R route through the projector contraction and the category gate is green at
N=3,4,5.

**STATUS — Phase 3 DONE (2026-07-18).** F/R now route through a projector contraction on the compact
`reduced_CGC` TensorMaps (`_Fsymbol`/`_Rsymbol`, `src/sector.jl`): the shared `d`/`c` line is traced
and divided by `dim`, all four OM legs open, result densified only at the tiny rank-2/4 output (never a
CGC). A `::Array{Float64,4}`/`{2}` type-assert is mandatory or `sectorscalartype` infers `Any`. Three
obstacles found + fixed, all written up in `phase3-derivation.typ`/`.pdf`:
1. **Tower self-consistency ⇒ dense gauge.** F/R fed back up the tower require `reduced_CGC(SU(N-1))`
   in the dense/GT gauge, else the level-N build stops intertwining. Added `_dense_gauge_transform`
   (`bootstrap.jl`): reproduce the dense `gaugefix!` convention (enumeration-order `qrpos!∘cref!`) on
   the c-HW slice, matched by GT-pattern identity, and apply the orthogonal `G` to the OM leg
   post-descent. Now `convert(Array, reduced_CGC) ≈ dense CGC` to ~1e-15 (sign / up to O(m)).
2. **`rightnull` tolerance.** The coset-raising `HW_eq` null singular value drifts to ~1e-11 for larger
   irreps → default `rightnull` returned empty and dropped channels (`only(blocks(CG_w))` threw, e.g.
   `15⊗15→1`). Fixed with `rtol=1e-9` (genuine couplings are O(1)). Was invisible in Phases 0–2
   (tested only on {4,6,10,20}; the gates request 15, 20⁺, 64 as intermediates).
3. **Descent precision.** The reduced descent is ~100× less accurate than dense at N=4 (isometry
   9e-15→1e-12) from SU(N-1) F-symbol *recoupling noise* injected into every `@tensor` (the per-shell
   solves are perfectly conditioned, κ=1 — a better solver buys nothing). Fixed by `_refine_reduced_CGC`:
   project the descent output onto the exact coset-intertwiner nullspace (assembled *compactly* over the
   isoscalar-factor coefficients, no densification) + polar (`tsvd`). Restores ~1e-14/machine precision
   at ~1s/CGC and stops the tower compounding its own error.

Result: **category gates green (0 failures) at N=3 and N=4** — fusiontensor↔F/R, F-unitarity, pentagon,
hexagon — with a fully compact construction (no dense CGCs). Bootstrap intrinsic gates green N=3,4,5;
N=5 F-vs-oracle ~4e-12. Full N=5 category gate not run (pentagon ~hours). Open follow-ups
(§ Phase 3 write-up): compact gauge-fix (**DONE, see below**); reference-state F/R to cut the `/dim`
trace accumulation (matters for N≥6 tower); full N=5 gate; matrix-free refinement for very large irreps.

**STATUS — compact gauge-fix DONE (2026-07-18).** The last CGC densification in the construction path
is gone. `_dense_gauge_transform` (`src/bootstrap.jl`) no longer calls `convert(Array, CGC)`; it reads
the c-HW slice compactly via a new recursive `_hw_slice(CGC, a, b, c)` — isoscalar factors from the
passed CGC's fusion-tree subblocks combined (Racah) with a one-level-down HW slice, bottoming out at a
tiny SU(2) CGC. Full GT patterns are rebuilt from `(reduced sector, SU(N-1) sub-pattern)` by a uniform
GT shift, and couplings are keyed by GT-pattern identity (so the §4 intra-sector ordering divergence is
irrelevant). The existing `gaugefix!`/`M\Q` are reused verbatim, so the gauge transform `G` is
byte-identical to the old dense path — hence `reduced_CGC` and the F/R are unchanged. Verified: the
compact slice equals the dense slice to ~1e-16, and `reduced_CGC` matches the independent dense LAPACK
solver on the c-HW slice (new `test/bootstrap.jl` testsets `compact _hw_slice == dense slice` and
`reduced_CGC vs dense CGC on c-HW slice`, green N=3,4). `src/bootstrap.jl` now has **zero**
`convert(Array, …)`; the only remaining densifications (`_Fsymbol`/`_Rsymbol`) act on the tiny rank-2/4
F/R output, never a CGC. All prior intrinsic gates green N=3,4,5; N=3 category gate re-confirmed green
(fusiontensor↔F/R 1280, F-unitarity 469, pentagon 625, hexagon 125; 0 failures).

---

## Phase 4 — Caching & cleanup

1. Restore `_get_CGC` (`clebschgordan.jl:25`): remove the dead early `return` or route the
   TensorMap CGCs through a dedicated cache. Decide the on-disk format — JLD2 can serialize
   `TensorMap`s; update `src/caching.jl` (`_key`, `cgc_cachepath`, `tryread`, `generate_CGC`).
2. Confirm the F/R LRU caches (`sector.jl` `FCACHE`/`RCACHE`) still behave.

**Done when:** repeated CGC/F/R calls hit the cache; `test/caching.jl` passes.

---

## Phase 5 — Tests & docs

1. New `test/bootstrap.jl` (wired into `test/runtests.jl`) asserting: generator commutators/Casimir
   (Phase 1), reduced-vs-dense CGC agreement + N=3 Kaeding values (Phase 2), and category gates via
   the existing `test/sectors.jl` machinery (Phase 3), across N=3,4,5.
2. README note on the reduced approach; link `research/theory-notes.md` and `research/pdf-guide.md`.

**STATUS — Phases 0–2 consolidated (2026-07-17).** `test/reduced.jl` + `test/reduced_CGC.jl` merged
into a single `test/bootstrap.jl` (`module BootstrapTests`, wired into `runtests.jl`), organized by
layer: (A) component units — `reduced_charge` partition, `reduced_space` multiplicity-free branching,
`_reduced_weight` round-trip, `reduced_basistransform` permutation, `highest_weight_projector`;
(B) coset generators — scalar subblocks, SU(3)-original + raw-SU(N) dense match, `−√2` golden value,
and a new **su(N) algebra gate** (reconstruct the full generators from the reduced coset operators →
commutators + quadratic Casimir eigenvalue `Σ_k λ_k(λ_k−2k+N+1)`); (C) `reduced_CGC` — isometry +
intertwiner gates, `dim(Vn)==Nsymbol`, degenerate-HW regressions, and a new **de Swart/Kaeding SU(3)
isoscalar-factor** cross-check. All green for N=3,4,5 standalone
(`julia --project=. -e 'include("test/bootstrap.jl")'`). Phase-2 math written up in
`phase2-derivation.typ`/`.pdf` (mirrors `phase1-derivation.typ`; CGC construction only). Remaining:
item 2 (README note); reduced-vs-dense *array* comparison intentionally omitted (basis-bridge
limitation — see Phase 2 STATUS) in favor of the intrinsic + literature gates. Category gates
(Phase 3) still run via `test/sectors.jl` on the dense F/R path.

---

## Sequencing & dependencies

```
Phase 0  ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4
(cleanup)   (generators) (reduced_CGC) (F/R)      (cache)
                                └────────────────▶ Phase 5 (tests, incremental per phase)
```
Phase 1 is the gate for everything; Phases 4/5 can proceed incrementally alongside 2/3.

## Cross-cutting conventions & gotchas (from the reading)

- **Isometry (standard-CG) normalization** throughout — `reduced_CGC` via `rightnull`/`isometry`.
  Closed loops therefore carry `dim` factors; keep every factor consistent with this convention (do
  **not** mix in the X-symbols Eq. 7 full-norm convention). (`weichselbaum2020`, `tensorkit2025`)
- **Index/fusion order:** TensorKit is codomain-first + left-to-right canonical; all `@tensor`
  contractions must match `reduced_CGC`'s `(a,b; c,μ)`. (`tensorkit2025` §4)
- **Multiplicity gauge:** multiplicity-free ⇒ overall sign only (`gaugefix!`); degenerate ⇒ SO(m)
  gauge freedom, pick and document a canonical basis. Sources differ legitimately (de Swart vs
  Pan-Draayer). (`pan1998`, `williams1996`, `alex2011` Eq. 37)
- **The `−i+1` GT shift** is embedded in the ladder formulas — the classic off-by-one source.
  (`molev2006`)
- **Frobenius-Schur χ_a=±1 and the B-symbol** enter every line-bend; wrong χ corrupts
  transposes/traces even when F/R look right. (`tensorkit2025` §5.2.5)
- **Inner multiplicity** (N≥3) needs a fixed deterministic ordering of degenerate-weight states.
  (`weichselbaum2012`, `alex2011` Fig. 2)
- **Label translation layer** when comparing to literature: (p,q)+dim names (Williams/Kaeding),
  Young diagrams [λ+μ,μ] (Pan-Draayer), vs the package's weights/Dynkin labels.

## Open questions / risks

1. **Coset-operator normalization/sign (Phase 1)** — the closed form fixes `r` up to one overall
   tensor-operator constant per fundamental; pin it with the reference-state cross-check + algebra
   gate (`generator-construction.md`). Also transcribe the Prop 2.4 norm ratio correctly (or take it
   from the cross-check).
2. **Outer-multiplicity gauge at N>3** — reduced-vs-dense comparison must be gauge-aware; decide
   whether to match Alex's dense gauge exactly or define an independent canonical one.
3. **HW-sector degeneracy** — verify the uniqueness assumption at `bootstrap.jl:141` for products
   where the HW fused sector carries multiplicity.
4. **Performance** — whether the SU(N-1)×U(1) block structure actually beats the dense solver at
   large irreps is the project's thesis; add a benchmark (vs `clebschgordan.jl`, and optionally the
   `alex2011` C++ / `su3lib2021` numbers) once N>3 works.
```
