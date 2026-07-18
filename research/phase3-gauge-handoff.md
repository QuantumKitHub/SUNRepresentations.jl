# Phase 3 handoff — reproducing the GT/dense gauge for reduced-native F/R

> **RESOLVED (2026-07-18) — superseded by `phase3-derivation.typ`/`.pdf`.** Phase 3 is complete: F/R
> route through the projector contraction on `reduced_CGC`, category gates green at N=3,4 (0 failures),
> fully compact (no dense CGCs). The gauge match recommended below (§6) was implemented as
> `_dense_gauge_transform`. Two additional issues not anticipated here were also required: a
> `rightnull` tolerance fix (empty-null-space channel drop, e.g. `15⊗15→1`) and an iterative
> refinement of the descent (`_refine_reduced_CGC`) to cure ~1e-12 recoupling-noise accumulation at
> N≥4. This file is kept for its detailed diagnostics; see the Phase 3 write-up for the final design
> and open follow-ups.

Self-contained brief for finishing Phase 3 (F/R symbols from `reduced_CGC` without densifying large
CGCs). Written 2026-07-17 after a deep investigation that reverted to the dense path. Read together
with `implementation-plan.md` (Phase 3), `generator-construction.md` (Phase 1), and
`phase2-derivation.typ` (Phase 2).

---

## 0. TL;DR

- **Goal:** compute SU(N) `Fsymbol`/`Rsymbol` by a projector contraction on the compact `reduced_CGC`
  `TensorMap`s — no full CGC densification in the F/R path.
- **Blocker found:** the whole lower-level machinery (`creation`/`annihilation`, the Molev coset
  scalars, the dense `CGC`) lives in the **GT basis**. `reduced_CGC(SU(N))`'s construction therefore
  needs TensorKit's SU(N−1) F/R in the **GT/dense gauge**. `reduced_CGC` itself comes out in a
  *different* gauge (per-channel sign + SO(m) + an internal-state ordering that
  `reduced_basistransform` does not exactly reproduce). Feeding that back breaks the level-N build.
- **Clean fix (the task):** make `reduced_CGC`'s `gaugefix!` reproduce the **dense sign/SO(m)
  convention**. Then `reduced_CGC == dense CGC` (up to an F/R-irrelevant internal reorder), projector
  F/R == dense F/R, the tower is self-consistent, and no large CGC is densified in the F/R path.
  `fusiontensor` stays the dense `CGC` (used only for explicit `convert(Array, ·)`, never in F/R).
- **Landed this session:** memoization of `reduced_CGC` and the coset operators
  (`REDUCED_CGC_CACHE`, `REDUCED_COSET_CACHE` in `src/caching.jl`) — a prerequisite for the recursive
  tower and worth keeping. Everything else reverted; all category gates green at N=3,4.

---

## 1. Where the gauge freedom lives

For a channel `a⊗b→c` there are two layers:

1. **Internal GT phases** — the individual basis vectors of each irrep. **Fixed**, no freedom: the
   Molev construction (`creation`/`annihilation` in `src/gtpatterns.jl`, using `signedroot`) gives
   every GT pattern a definite phase. The coset ladder operators live in this layer.
2. **The coupling gauge** — the *only* freedom: an overall **sign** per multiplicity-free channel, or
   an **O(m)** rotation of the `m` outer-multiplicity copies per degenerate channel. `reduced_CGC`
   fixes this via `gaugefix!` on its highest-weight block.

`reduced_CGC` and the dense `CGC` therefore differ *only* in layer 2 (plus an internal ordering, see
§4). Confirmed empirically: after undoing `reduced_basistransform`, `convert(Array, reduced_CGC)`
equals the dense `CGC` to ~1e-15 for multiplicity-free products (SU3 and SU4, incl. `15⊗4→36`),
differing only by a per-channel sign; degenerate `8⊗8→8` (N=2) differs by an SO(2) rotation.

---

## 2. How the dense/GT gauge is defined (the convention to reproduce)

Source: `src/clebschgordan.jl`.

- `highest_weight_CGC` (`:71`) solves the HW nullspace. The solution matrix `solutions` has
  **rows = fused `(m₁,m₂)` states at weight `HW(s₃)`**, ordered by the enumeration
  `for (m₁,pat₁) in enumerate(basis(s₁)), for m₂ in ...` — i.e. **m₁ outer over `basis(s₁)`, m₂
  inner over `basis(s₂)`**. Columns = the outer multiplicity.
- `solutions = gaugefix!(solutions)` (`:112`), where
  `gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))` (`:50`):
  - `cref!` (`:240`) = column-reduced echelon form, scanning **rows top-to-bottom** (= the
    enumeration order), max-abs pivot per row.
  - `qrpos!` (`:230`) = QR with **positive R-diagonal**.
- The HW coupling is stored at `CGC[m₁m₂, d₃, α]` with `d₃ = dim(s₃)` = the **last** basis index
  (`:118`).

**Basis order gotcha:** `basis(a)` runs **lowest-weight-first** — `basis(3)[1] = GT(1,0,0,0,0,0)`
while `highest_weight(3) = GT(1,0,0,1,0,1) = basis(3)[end]`. So "first in enumeration order" is a
**low-weight** coupling, *not* the highest-weight one.

**Net rule (multiplicity-free):** the sign is fixed so that the **first nonzero coupling coefficient,
in `basis(a)⊗basis(b)` enumeration order, is positive.** Verified: `3⊗3̄→1` has first coupling
`m₁ = basis(3)[1] = (1,0,0,0,0,0)`, `m₂ = (1,1,0,1,1,1)`, coefficient `+0.5774`.

**Degenerate (m>1):** the multiplicity basis is the `cref!+qrpos!` canonical form in that same row
order (RREF + positive-diagonal QR).

---

## 3. How `reduced_CGC`'s gauge currently differs

Source: `src/bootstrap.jl`, `reduced_CGC`/`_reduced_CGC` (`:168`/`:174`).

- HW block: `charge, bl = only(blocks(CG_w))` (`:190`), then `block(CGC, charge) .= gaugefix!(bl)`
  (`:194`). Here `bl`'s **rows are the reduced HW-sector coupling multiplicity** (reduced-sector
  order) — a *coarser and reordered* basis than dense's `(m₁,m₂)` rows.
- Same `gaugefix!` (`cref!+qrpos!`) but applied in that different row basis ⇒ a different overall
  sign / SO(m) per channel than dense.

Measured per-channel signs (SU3, `convert(Array, reduced_CGC)[invperm(rbt)]` vs dense): most `+`;
`−` for `3⊗3̄→1`, `3⊗8→3`, `3̄⊗3̄→3`, `8⊗8→10̄`; `8⊗8→8` (N=2) differs by SO(2).

---

## 4. The `reduced_basistransform` ordering divergence (important, subtle)

`reduced_basistransform(a) = sortperm(basis(a); by = x -> (reduced_charge(x), U1Irrep(rowsum(x, N-2))))`.

It groups GT patterns by reduced sector correctly, but the **secondary key `rowsum(·, N-2)` does not
fully match TensorKit's `convert`-order internal ordering within multi-dimensional sub-sectors.** For
`maxdeg==1` / small sub-sectors it happens to match (bridge probe: ~1e-15). For larger SU(N−1)
sub-sectors it diverges. This is why:

- setting `fusiontensor = convert(Array, reduced_CGC)` puts the internal basis in an order that
  disagrees with the GT-basis machinery, and
- the reference-state **extraction** (project the raw generator onto `convert(Array, unit_subblock)`)
  **fails**: `A_raw` (in `reduced_basistransform` order) and `convert(Array, unit_subblock)` (in
  TensorKit's convert order) misalign → wrong scalars → `SingularException` in `reduced_CGC`'s
  descent solve.

Corollary: do **not** try to make `fusiontensor = densify(reduced_CGC)` unless you also make
`reduced_basistransform` exactly equal TensorKit's `convert` order (sort by the *full recursive GT
position within a sector*, not `rowsum(·, N-2)`). The recommended fix in §6 avoids this entirely.

---

## 5. Why the tower requires the dense gauge (the mechanism)

- `reduced_raising_operators`/`reduced_lowering_operators` (`src/bootstrap.jl`) write Molev scalars
  into fusion-tree subblocks `g[f₁,f₂]`, where `f₁` couples **fund/antifund(SU(N−1)) ⊗ ν′ → ν**
  (always multiplicity-free). The scalar is only correct if TensorKit's SU(N−1) CGC for that coupling
  has the sign the Molev formula assumes = the **GT/dense** sign.
- `reduced_CGC(SU(N))`'s construction (`@tensor` over SU(N−1)⊠U(1)) consumes SU(N−1) **F/R**, and the
  coset scalars implicitly assume the GT gauge. So `reduced_CGC(SU(N−1))` must be in the dense gauge,
  or the level-N build stops intertwining. Measured failure with a non-dense gauge:
  `‖(J⁺_a⊗1+1⊗J⁺_b)C − C J⁺_c‖ ≈ 2.0` (should be ~1e-13); F-unitarity fails at N=4.
- **Crucial for savings:** the `reduced_CGC` build uses SU(N−1) **F/R** (via `@tensor`), *not*
  `fusiontensor`. So F/R via a projector on `reduced_CGC` (no densify) is enough for the whole tower;
  `fusiontensor`/dense `CGC` is only needed for explicit `convert(Array, ·)`.

---

## 6. The fix to implement (recommended)

**Make `reduced_CGC`'s `gaugefix!` reproduce the dense sign/SO(m) convention of §2**, then compute
F/R by a projector contraction on `reduced_CGC`. Steps:

1. **Gauge-match `reduced_CGC`.** At the HW block (`bootstrap.jl:190-194`), instead of
   `gaugefix!(bl)` in reduced-sector order, gauge-fix in **GT enumeration order**:
   - Densify *only the small HW-weight coupling block* — the fused states at weight `HW(c)` (this is
     tiny compared with the full CGC), in `basis(a)⊗basis(b)` enumeration order.
   - Apply the identical `gaugefix! = qrpos!∘cref!` there (this *is* the dense convention).
   - Map the resulting sign/SO(m) back onto the reduced HW block and seed the descent.
   The recursion is linear per multiplicity column, so equivalently: build `reduced_CGC` as now, then
   fix each column's overall sign / SO(m) from the densified HW-weight block. Fund couplings are
   multiplicity-free ⇒ the tower-critical part is a single sign per channel.
2. **F/R via projector on `reduced_CGC`.** Reinstate the reduced contraction (the reverted `f08dd65`
   shape) and close the shared internal line through `highest_weight_projector` (or full trace
   `/dim(d)` first as a known-correct stage), keeping the four OM legs open:
   ```julia
   A = reduced_CGC(a,b,e); B = reduced_CGC(e,c,d); C = reduced_CGC(b,c,f); D = reduced_CGC(a,f,d)
   @tensor F[-1 -2; -3 -4] := A[1 2 3; -1]*B[3 4 6; -2]*conj(C[2 4 5; -3])*conj(D[1 5 6; -4]) / dim(d)
   ```
   **Re-verify the sign** vs the dense F once `reduced_CGC` is in the dense gauge: the sign flips seen
   earlier (conjugate irreps) were traced to `reduced_CGC`'s gauge, *not* provably to `conj`'s
   Frobenius–Schur handling — with the gauge matched they may resolve. If a residual FS/duality sign
   remains on the `conj(TensorMap)` legs, compute via adjoints (`C'`, `D'`) or an explicit B-symbol
   bend, and pin it against the dense oracle at N=3 (`3⊗3̄` conjugate channels are the discriminating
   cases).
3. **Type-stability (mandatory).** `Fsymbol`/`Rsymbol` must be **inferrable as `Array{Float64,4}` /
   `Array{Float64,2}`**. `sectorscalartype(I)` uses `Core.Compiler.return_type(Fsymbol, NTuple{6,I})`
   then `eltype` (`TensorKitSectors/src/sectors.jl:123-138`); if that infers `Any` (because
   `reduced_CGC` is cached as `Any`), TensorKit does `zeros(Any, …)` → `MethodError: zero(Type{Any})`
   when building tensors over `ProductSector{SUNIrrep{N-1},U1Irrep}`. Fix with a **`::Array{Float64,4}`
   type assertion** on the return (a `convert(...)` does NOT propagate through inference; the `::`
   typeassert does).
4. Keep `fusiontensor = CGC(Float64, …)` (dense) — consistent with the dense-gauge F/R and used only
   for explicit densification.

---

## 7. Verification ladder (do in this order)

- **Intrinsic** (`test/bootstrap.jl` `check`): `‖C†C − 1‖ < 1e-10` and coset intertwiners
  `‖(J_a⊗1+1⊗J_b)C − C J_c‖ < 1e-10` for `J = J±`, at N=3,4,5 — confirms the gauge-matched
  `reduced_CGC` still builds correctly.
- **reduced-vs-dense (now exact):** `convert(Array, reduced_CGC(a,b,c))[invperm(rbt),…] ≈ dense
  CGC(a,b,c)` **including sign** for multiplicity-free products; up to SO(m) for degenerate. This is
  the direct check that the gauge match worked.
- **F vs dense oracle at N=3:** compare projector `Fsymbol(a,b,c,d,e,f)` to the dense reference-state
  formula on `CGC(Float64,…)` (the discriminating cases are conjugate channels like
  `a=b=c=3̄, d=1`). Must match to ~1e-8.
- **Category gates** (`test/sectors.jl`): fusiontensor↔F/R, F-unitarity `F'F≈1`, pentagon, hexagon at
  N=3,4,5 and the product sector `SUNIrrep{3}⊠SUNIrrep{3}`. Gate one N at a time (temporarily set
  `sectorlist=(SUNIrrep{3},)`, then 4, then 5).

Baseline (measured this session, dense path): all category gates green at N=3,4.

---

## 8. Key code locations

| item | location |
|---|---|
| dense `gaugefix!` = `qrpos!∘cref!` | `src/clebschgordan.jl:50`, `:230`, `:240` |
| dense HW solve + gaugefix + storage | `src/clebschgordan.jl:71-123` (`:112`, `:118`) |
| `reduced_CGC` / `_reduced_CGC` | `src/bootstrap.jl:168` / `:174` |
| reduced HW block gaugefix (change here) | `src/bootstrap.jl:190-194` |
| coset ladder ops (Molev closed form) | `reduced_raising_operators`/`reduced_lowering_operators`, `src/bootstrap.jl` |
| coset magnitude `_raise_mag2` | `src/bootstrap.jl` |
| `reduced_charge`/`reduced_space`/`reduced_basistransform`/`_reduced_weight` | `src/bootstrap.jl:8/17/28/…` |
| `highest_weight_projector` | `src/bootstrap.jl` |
| `fusiontensor`, `_Fsymbol`, `_Rsymbol`, `FCACHE`/`RCACHE` | `src/sector.jl` |
| caches (memoization, landed) | `src/caching.jl` `REDUCED_CGC_CACHE`, `REDUCED_COSET_CACHE` |
| `sectorscalartype` (type-inference trap) | `TensorKitSectors/src/sectors.jl:123-138` |
| raw generator reference (for intrinsic checks) | `test/bootstrap.jl` `raw_raising_ref`/`raw_lowering_ref` |

---

## 9. Environment & how to run

- `julia --project=.` — local `Manifest.toml` resolved (TensorKit 0.14.11, Julia 1.12.6). No scratch
  env needed.
- Reduced/bootstrap tests: `julia --project=. -e 'include("test/bootstrap.jl")'`.
- Category gates: `Pkg.test()`, or an isolated driver that loops `for I in (SUNIrrep{3},)` over the
  `test/sectors.jl` testsets (fusiontensor↔F/R, F-unitarity, `pentagon_equation`,
  `hexagon_equation`) for fast per-N iteration.

## 10. Diagnostic recipes that were useful (recreate as needed)

- **Bridge/sign probe:** for `(a,b,c)`, compare `convert(Array, reduced_CGC(a,b,c))[invperm(rbt),…]`
  to `Array(CGC(Float64,a,b,c))` per multiplicity column; classify each as `+` / `−` / SO(m).
- **Intrinsic intertwiner:** `norm((Jpa⊗1+1⊗Jpb)*C − C*Jpc)/max(...)` with the reduced coset ops.
- **Inference check:** `Core.Compiler.return_type(Fsymbol, NTuple{6,SUNIrrep{3}})` and
  `sectorscalartype(SUNIrrep{3})` must be concrete (`Array{Float64,4}` / `Float64`), not `Any`.
- **F-vs-oracle:** dense reference-state F from `CGC(Float64,…)` using
  `A=CGC(a,b,e); B=CGC(e,c,d)[:,:,1,:]; C=CGC(b,c,f); D=CGC(a,f,d)[:,:,1,:];`
  `@tensor F[-1,-2,-3,-4] := conj(D[1,5,-4])*conj(C[2,4,5,-3])*A[1,2,3,-1]*B[3,4,-2]`.
