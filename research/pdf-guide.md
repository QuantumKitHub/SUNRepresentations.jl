# PDF guide — what's in each paper and where to look

A navigator for the PDFs in `pdfs/`. For each paper: a section→page map, the load-bearing
results (with equation/page pointers), and **how it feeds the implementation** (phases from the
roadmap: **P1** generators in the SU(N-1)×U(1) basis, **P2** CGC equations as TensorMap equations,
**P3** F/R-symbols). Annotated citations are in `references.md`; the *why it's exact* math is in
`theory-notes.md`. Page numbers are the paper's own printed numbers unless noted.

## Quick index

| PDF | Paper | Gives you | Phase |
|-----|-------|-----------|-------|
| `arXiv-1009.0437.pdf`      | Alex et al. 2011 | The dense algorithm this package reimplements: GT patterns, ladder matrix elements, HW-nullspace, lowering recursion, outer-multiplicity gauge | P1, P2 (ground truth) |
| `arXiv-1202.5664.pdf`      | Weichselbaum 2012 (QSpace) | The "reduced ⊗ CGC" tensor decomposition; MW-lowering CGC construction; outer/inner multiplicity bookkeeping | P1, P2 (design) |
| `arXiv-1910.13736.pdf`     | Weichselbaum 2020 (X-symbols) | How F/R & 3n-j data come from CG tensors; the "projector not trace" recipe | P3 (direct) |
| `arXiv-2508.10076.pdf`     | Devos & Haegeman 2026 (TensorKit.jl) | The target interface: sectors, fusion trees, topological data, how a TensorMap is stored & what F/R must satisfy | all (conformance) |
| `arXiv-math_0211289.pdf`   | Molev 2006 | Rigorous GT bases for gl_n: branching, exact Chevalley matrix elements, norms — underpins `gtpatterns.jl` | P1 |
| `arXiv-nucl-th_9502037.pdf`| Kaeding 1995 | Tabulated SU(3) isoscalar factors → **N=3 validation numbers** | P2 (validation) |
| `arXiv-hep-th_9509167.pdf` | Williams 1996 | SU(3) ISF recursion relations + degeneracy seeding | P2 |
| `arXiv-quant-ph_9704015.pdf`| Pan & Draayer 1998 | The outer-multiplicity gauge (SO(m)) and a canonical fix — the `Vn` authority | P2 (multiplicity) |

---

## 1. Alex, Kalus, Huckleberry, von Delft (2011) — `arXiv-1009.0437.pdf` (30 pp)
*J. Math. Phys. 52, 023507.* The algorithm `src/clebschgordan.jl` reimplements; C++ source in App. D.

**Where to look**
- §IV–V (p.5–6): sl(N,ℂ) basis (Eqs. 16–18) and GT-pattern labelling (Eqs. 19–22).
- §VII (p.7–9): **raising/lowering matrix elements** — Eq. 28 (lowering), Eq. 29 (raising).
- §X (p.10–12): **highest-weight = null space** of raising ops (Eqs. 35–36); outer-multiplicity resolution by RREF + Gram-Schmidt (Zaránd gauge, Eq. 37).
- §XI (p.12–13): **lower-weight recursion** (Eq. 40), overdetermined least-squares.
- §VIII (p.9–10): which irreps appear (Littlewood-Richardson via GT, Eq. 31); App. C (p.16–19): integer indexing.

**Key facts**
- HW null-space dimension = outer multiplicity `N^{S''}_{S,S'}` (one solution per α). Solved by SVD; the RREF+Gram-Schmidt normal form is *optional* in their code (p.29) — so the dense solver's CGCs may be orthonormal-but-not-canonical.
- Inner multiplicity >1 for N≥3: lowering paths `J⁻_l J⁻_{l'}` don't commute, so the recursion must sum over *all* parents (Fig. 2, p.8), not one path.
- Sub-irrep i-weights are **not** normalized (p.6) — needed to distinguish repeated SU(N-1) irreps.

**Use for** — P1: Eqs. 28/29 are exactly the ladder matrix elements to populate generator TensorMaps; validate `creation`/`annihilation` against them element-wise. P2: Eq. 36 (HW nullspace) and Eq. 40 (lowering) are the equations `reduced_CGC` solves; the branch's `_nullspace!` / `rightnull` mirrors their SVD. **Ground-truth check:** `convert(Array, reduced_CGC)` ≈ `CGC(...)` up to the multiplicity gauge — beware their gauge may differ from TensorKit's canonical basis.

## 2. Weichselbaum (2012), QSpace — `arXiv-1202.5664.pdf` (Ann. Phys. 327, 2972)
The conceptual parent. Largest worked non-abelian example is **SU(3)/Sp(6)** — it does *not* carry out an explicit SU(N>3) construction, and it labels by **weights, not GT patterns / isoscalar factors**. So it's the design blueprint for the algebra + multiplicity bookkeeping, but its CGCs are in the full weight basis (a change of basis away from the branch's SU(N-1)×U(1) isoscalar layout).

**Where to look**
- §II.A–C (p.5–8): the QSpace = **reduced-block ⊗ per-symmetry CGC tensor** (Eqs. 3–5, 8) — the same object as the branch's TensorMap-of-CGCs. Wigner-Eckart Eq. 4; general form Eq. A43 (p.35).
- **App. B.1 (p.46–47, Fig. 13): the load-bearing CGC-construction algorithm** — MW seed → apply simple lowering ops → orthonormalize → sort by descending z-labels → next seed. No Casimirs needed.
- App. A.4 (p.31): explicit SU(N) generators — `E_{ij}`, the r=N−1 diagonal Cartan/z-operators (Eq. A28b), simple-root raising set {S₁₂,…,S_{N-1,N}} (Eq. A29).
- §II.C + App. A.5–A.6 (p.8–9, 33–34): outer multiplicity as **separate records with identical labels + orthogonal CGC spaces** (Eqs. A38–A39), *not* a 4th tensor index.

**Use for** — P1: adopt the Eq. A27–A29 generator basis; the U(1) charge is one Cartan combination, the SU(N-1) block is the `i,j<N` generators. P2: Fig. 13 is a direct template for HW+lowering; **its outer-multiplicity design (orthogonal CGC copies) maps cleanly onto TensorKit's multiplicity leg** — this is the key transferable decision. Inner multiplicity needs a *fixed deterministic ordering* of degenerate-weight states (subtle correctness trap). P3: it does **not** do F/R (states 3n-j symbols are generally unknown) — it contracts CGC networks numerically; the "fully contracted → identity/scalar" invariant (p.6–7) is a good test.

## 3. Weichselbaum (2020), X-symbols — `arXiv-1910.13736.pdf` (Phys. Rev. Research 2, 023385)
The Phase-3 reference: how F/R and general 3n-j symbols come from CG tensors.

**Where to look**
- §I.B–C (p.3–4): tensor = ⊕ reduced ⊗ CGT (Eq. 2); outer-multiplicity index sits **on the CGT node, not a leg** (Eqs. 4–5).
- §II.B (p.5, **Eq. 7**): CGT normalization `Tr(C_{q,μ} C†_{q,μ'}) = δ_{μμ'}` (full contraction) — explicitly **differs from standard Wigner CG** normalization, where the same trace gives `dim(fused irrep)`. **This dimension factor is exactly the origin of `/dim(d)`.**
- §II.C (p.6, Eq. 8): tracing all-but-one leg of a contracted CGT network gives `(1/|q_i|)·identity`.
- §III.A (p.8–9, **Eqs. 15–16**): the X-symbol = contract two CGTs, then project onto the orthonormalized composite `E` via `E†E=1` (a **projector**, not trace-and-divide).
- §III.B (p.10, Eq. 17): the **6j-symbol = four CG3s, six contracted lines**; for SU(N≥3) it's a **rank-4 tensor over the four OM indices** (F- and R-symbols are such networks).

**Use for** — P3, directly. The current `_Fsymbol` (`src/sector.jl:86`) closes the 6j loop over all six irrep lines and divides by `dim(d)`. Because `reduced_CGC` is built by `rightnull` it's an **isometry** (standard-CG norm), so by Schur `Tree_R†∘Tree_L = F ⊗ 1_d`; tracing `d` gives `dim(d)·F`, hence `/dim(d)` is *numerically correct but wasteful*. **The "replace trace with projector" fix:** contract the shared `d` (resp. `c`) line through `highest_weight_projector(Vd, d)` (`src/bootstrap.jl:106`) onto its HW sub-block and normalize by that block's `blockdim` (or evaluate on one reference state, no division) — cheaper and avoids summing the whole line. Keep all four OM legs open (already correct). SU(2) has no rank-3 OM ⇒ scalar F/R ⇒ good regression case; OM (rank-4 F) only appears for N≥3, exactly where projector-vs-trace correctness matters.

## 4. Devos & Haegeman (2026), TensorKit.jl — `arXiv-2508.10076.pdf`
The interface this package must satisfy. (First author is the repo owner.)

**Where to look**
- §2.1–2.2 (p.7–11): TensorMap = matrix `t̃` between fused domain/codomain (Eq. 9); index order **codomain-first**, then domain.
- §4.1 (p.33–35): **fusion trees**; splitting/fusion tensors `X^{ab}_{c;μ}` are CGCs (Eqs. 80–81); Wigner-Eckart Thm 1 (Eqs. 84–87); block-by-coupled-charge storage (Eq. 94). **TensorKit never needs explicit CGC values — only orthonormality/completeness (Eqs. 83, 138–140).**
- §5.2 (p.49–53): the topological data a sector must implement — N-symbols (Eq. 134–136), quantum dims (Eq. 137), **F-symbols + pentagon (Eq. 147)**, duality/Frobenius-Schur (Eqs. 148–155), **R-symbols + hexagons (Eqs. 158–159)**.
- §5.5.1 (p.55–56): **Rep^G recipe** — given fusion rules + CGCs, get F/R by *projecting* onto the fusion-tensor basis (Eqs. 165–166); SU(2): F = 6j (Eq. 163), R = sign (Eq. 164).

**Use for** — all phases (conformance). A `reduced_CGC` is a splitting tensor and must be an **isometry** (Eqs. 138–140) — build with `isometry`/`rightnull`. F/R follow Eqs. 165–166 and must satisfy pentagon/hexagon (reuse as tests). For SU(N≥3) fusion multiplicities >1, so the μ,ν,κ,λ OM indices are load-bearing. **Warnings:** match codomain-first + left-to-right canonical fusion order or contractions silently pick the wrong basis; Frobenius-Schur χ_a=±1 and the B-symbol enter every line-bend (get them wrong and transposes/traces corrupt even if F/R look right); reduced coefficients take *linear combinations* under permutation (not mere reshuffles). Reusable utilities: `fuse`, `block`, `isometry`, `rightnull`, adjoint.

## 5. Molev (2006) — `arXiv-math_0211289.pdf` (Handbook of Algebra 4)
Rigorous GT-basis math for classical Lie algebras; **§2 (gl_n, type A) is the relevant part**. (Printed page = PDF page − 2.)

**Where to look**
- Thm 2.1 (p.8): branching gl_n↓gl_{n-1}, multiplicity-free, betweenness (2.2) — the SU(N)↓SU(N-1) chain.
- **Thm 2.3 (p.9): the classical GT matrix-element formulas** — Cartan (Eq. 2.5), raising (2.6), lowering (2.7), in shifted variables `l_{ki}=λ_{ki}−i+1`. Only simple/Chevalley generators get closed forms; others via commutators.
- Prop 2.4 (p.10): orthonormal basis `ζ_Λ = ξ_Λ/‖ξ_Λ‖` ⇒ ladder matrices become **signed square roots** (the code's `signedroot`).
- §2.1/§2.3 (p.10–17): lowering/raising as explicit algebra elements `z_{in}, z_{ni}`; they commute among themselves (Prop 2.12) ⇒ type-A ordering is unambiguous. Thm 2.11 (p.14): the branching multiplicity space is generated by the last-level lowering operators.

**Use for** — P1. Confirms `src/gtpatterns.jl` `creation` (lines 143–169) *is* Eq. 2.6 in the orthonormal form: the numerator = level-(l+1)·level-(l−1) factors, denominator = level-l factors, `signedroot` = the symmetric √. For general N, Thm 2.3 already gives all simple-root generators; the SU(N-1)×U(1) block = fix the top row (U(1)/trace + SU(N-1) irrep μ), the k=N−1 lowering ops move between μ-blocks (Thm 2.11), lower-k generators are block-diagonal = the SU(N-1) generators. **Warnings:** Molev's "n" = matrix size (gl_n = U(N)); project the Cartan to traceless differences for SU(N) (as `Zweight` does). The `−i+1` shift is baked into every formula — off-by-one here is the classic bug.

## 6. SU(3) isoscalar-factor trio — the N=3 ground truth + the multiplicity authority
All three work in SU(3)⊃U(2)=SU(2)×U(1); their ISFs/reduced Wigner coefficients **are** the branch's reduced blocks. Kaeding & Williams share the **de Swart** phase convention (Williams' algorithm generated Kaeding's tables); Pan-Draayer use a *different* canonical gauge.

### 6a. Kaeding (1995) — `arXiv-nucl-th_9502037.pdf` (At. Data Nucl. Data Tables 61, 233)
Mostly tables. §1 (p.1–2): conventions — Condon-Shortley within isomultiplets, de Swart between them; full CGC = **ISF × SU(2)-CGC** (Eq. 2); entries are `±√C`. Table 1 (p.6) lists tabulated products (3⊗3, 3̄⊗3, 6⊗3, 8⊗3, 10⊗3, 15⊗3, … through 15'⊗15'); **the only multiplicity-2 case is 15⊗8**; 8⊗8, 10⊗8, 10⊗10, 27⊗8 are *not* reproduced (see de Swart / McNamee-Chilton).
**Use for** — P2 validation at N=3. Cleanest targets: the **multiplicity-free** products (only sign ambiguity, fixed by Eq. 1). Match conventions: SU(2) leg in Condon-Shortley; overall HW sign per Eq. 1.

### 6b. Williams (1996) — `arXiv-hep-th_9509167.pdf` (J. Math. Phys. 37, 4187)
§II (p.3–4): ISF factorization `C = C^{SU2}·F` (Eq. 11), with I, I_z, Y from (k,l,m) (Eqs. 5–7). §III (p.4–5): degeneracy formula (Eq. 13); proves **Braunschweig's conjecture** (enough SHW WCGs to seed multiplicity). **Recursion relations** (the deliverable): four-term ISF recursions Eqs. 32 & 34 (from `V̂₊|SHW⟩=0`, `Û₋|SHW⟩=0`), step-down Eqs. 35–36. §IV (p.9–10): degeneracy resolution by **Kronecker seed + Gram-Schmidt** (order-dependent → *a* gauge, not canonical).
**Use for** — P2. The recursion is the same ladder-operator mechanism the branch uses; consistent with Kaeding for validation. The Gram-Schmidt multiplicity recipe is one concrete `Vn` choice, but its non-uniqueness is what Pan-Draayer formalizes.

### 6c. Pan & Draayer (1998) — `arXiv-quant-ph_9704015.pdf` (J. Math. Phys. 39, 5642)
**The outer-multiplicity authority.** §II (p.3–9): two valid resolutions of an m-fold degeneracy are related by **SO(m)** (real case; SU(m) complex) — so the reduced blocks in the degenerate sector are *gauge-dependent*, with **no gauge-independent "correct" value**. A canonical fix: an SO(m) rotation `Y` making the RWC matrix **lower-triangular** (Eqs. 2.12–2.21) + positive phase (Eq. 2.19); coincides with Hecht / Le Blanc-Rowe. §III: builds multiplicity RWCs from multiplicity-free U(n)⊃U(n-1) RWCs via recoupling + complementary group (U(4) for SU(3)). §IV (p.20–22): the canonical RWCs **do not** obey the 1↔2 exchange symmetry that de Swart/Williams ones do; explicit 2×2 rotation Eq. 4.13; Tables I–II (p.29–30) print both gauges side by side.
**Use for** — P2 multiplicity, at N>3. Model `Vn` as a genuine **gauge choice**: pick a deterministic canonical basis (lower-triangular + positive phase) and state it. Expect Kaeding's one degenerate case (15⊗8) to be in the *de Swart* gauge, not Pan-Draayer's; validate degenerate sectors up to an orthogonal transform, not element-wise.

---

## Reading path by implementation phase

- **P1 — generators in the SU(N-1)×U(1) basis (TensorMaps).** Molev §2 (exact Chevalley matrix elements, Thm 2.3) + Alex §IV, VII (Eqs. 16–18, 28–29) for the formulas; QSpace App. A.4 (Eqs. A27–A29) for the generator basis and how to split U(1) vs the SU(N-1) block. Validate against `src/gtpatterns.jl`.
- **P2 — CGC equations as TensorMap equations.** Alex §X–XI (HW nullspace Eq. 36 + lowering Eq. 40) and QSpace Fig. 13 for the algorithm; TensorKit §4.1/§5.5.1 for the isometry/normalization the result must satisfy. For **outer multiplicity**: QSpace §II.C (orthogonal-copies design) + Pan-Draayer §II (SO(m) gauge, canonical fix). Validate N=3 against **Kaeding** (multiplicity-free) and Williams; check degenerate sectors up to SO(m).
- **P3 — F/R-symbols.** X-symbols §II.B–III (Eq. 7 normalization, Eq. 16 X-symbol, Eq. 17 6j) for the "projector not trace" fix to `src/sector.jl`; TensorKit §5.2/§5.5.1 (Eqs. 165–166) for the projection recipe and pentagon/hexagon tests.

## Conventions & gotchas (cross-cutting)

- **Normalization.** `reduced_CGC` is an **isometry** (standard-CG norm), so closed loops carry `dim` factors (the origin of `/dim(d)`). X-symbols' Eq. 7 (full-norm-1) would remove them but is *not* what the code produces — keep every factor consistent with the isometry convention.
- **Index/fusion order.** TensorKit is codomain-first + left-to-right canonical fusion; leg order/arrows in every `@tensor` must match how `reduced_CGC` orders `(a,b; c,μ)`.
- **Multiplicity gauge.** Multiplicity-free sectors: only an overall sign (fix once, e.g. `gaugefix!`). Degenerate sectors: an SO(m) gauge freedom — pick a canonical basis and be explicit; different sources (de Swart vs Pan-Draayer) are legitimately different numbers.
- **The `−i+1` GT shift** (Molev) is embedded in the ladder formulas — the most common off-by-one source.
- **Frobenius-Schur / duality** (TensorKit §5.2.5): χ_a=±1 and the B-symbol enter every line-bend; wrong χ corrupts transposes/traces even when F/R look right.
- **Label translations.** Williams/Kaeding use (p,q)+dim names; Pan-Draayer use Young-diagram [λ+μ,μ]; the package uses Young-row weights / Dynkin labels — keep a translation layer when comparing.
