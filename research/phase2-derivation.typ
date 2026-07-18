#set page(margin: 2.2cm, numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show link: set text(fill: blue)
#set math.equation(numbering: "(1)")

#let box-eq(body) = align(center, block(
  stroke: 0.6pt + gray, inset: 9pt, radius: 4pt, width: 92%,
)[#body])

#align(center)[
  #text(15pt, weight: "bold")[Reduced SU($N$) Clebsch–Gordan coefficients in an SU($N-1$)$times$U(1) basis]
  #v(2pt)
  #text(10pt)[Phase 2 derivation — `SUNRepresentations.jl`, branch `reduced`]
]

#v(4pt)

This note derives `reduced_CGC(a, b, c)`: the SU($N$) Clebsch–Gordan coefficient (CGC) for
$a times.o b arrow.r c$, stored as a TensorKit `TensorMap` over SU($N-1$)$times$U(1) sectors.
Its blocks are the SU($N$)$supset$SU($N-1$)$times$U(1) *isoscalar factors*. The construction reuses
the coset ladder operators built in Phase 1 (@sec-phase1); it is the reduced-basis mirror of the
dense two-stage highest-weight/lowering algorithm @alex2011. Sources are collected in @sec-sources.

= CGCs as `TensorMap`s <sec-setup>

An SU($N$) irrep restricts along the chain SU($N$) $supset$ SU($N-1$)$times$U(1); its Gelfand–Tsetlin
states carry a reduced charge $(nu, Y)$ (`reduced_charge`), and `reduced_space(a)` bins them into a
graded space $V_a$ over the product sector $nu ⊠ Y$ (@sec-phase1). The branching is multiplicity-free,
so every sector of $V_a$ has degeneracy 1.

Write $G = "SU"(N)$ and $H = "SU"(N-1) times "U"(1)$. By the *Racah factorization lemma*
@deswart1963 @molev2006, an SU($N$) CGC factorizes along the chain $G supset H$ into an $H$-CGC times
an *isoscalar factor* (reduced coupling coefficient) that is independent of the $H$-internal labels:
$
  lr(chevron.l a med m_a\; b med m_b mid(|) c med m_c chevron.r)_G
    = lr(chevron.l alpha_a\; alpha_b mid(|) alpha_c chevron.r)_H dot.c
      lr([a b; c])_(alpha_a alpha_b alpha_c),
$
where $m = (alpha, ...)$ splits into the $H$-charge $alpha$ and the internal $H$-labels. This is
exactly how TensorKit @tensorkit2025 stores a symmetric map: the coefficient of one
splitting/fusion-tree pair — the subblock `C[f₁, f₂]` — is the isoscalar factor, while the $H$-CGC is
supplied by the tree. So the SU($N$) CGC *is* the `TensorMap`

#box-eq[
  $ C : V_a times.o V_b arrow.l V_c times.o V_n, wide
    dim V_n = upright(N)_(a b)^c, $
]

where the extra leg $V_n$ (a trivial $H$-sector) carries the *outer multiplicity*
$upright(N)_(a b)^c = $ `Nsymbol(a,b,c)`
(the number of independent couplings $a times.o b arrow.r c$), a trivial SU($N-1$)$times$U(1)
sector. Following @weichselbaum2012[§II.C] the multiplicity is a genuine leg — orthogonal CGC copies
sharing labels — matching TensorKit's multiplicity index. Throughout, $C$ is normalized as an
*isometry* (standard-CG convention): $C^dagger C = bold(1)$ on $V_c times.o V_n$
@tensorkit2025 @weichselbaum2020.

= Highest-weight coupling <sec-hw>

The highest-weight (HW) coupling is annihilated by all raising operators. Writing $J^(+)_a$ for the
combined coset raising operator of $a$ (Phase 1; the SU($N-1$)$times$U(1) raisings act within sectors
and are automatic), the HW block $C_"hw"$ solves

#box-eq[
  $ (J^(+)_a times.o bold(1) + bold(1) times.o J^(+)_b) med C_"hw" = 0 $ <eq-hw>
]

restricted to the fused HW reduced sector — the sector of `reduced_charge(highest_weight(c))` inside
$V_a times.o V_b$, isolated by `highest_weight_projector` @alex2011[Eq. 36]
@weichselbaum2012[Fig. 13]. In code (`bootstrap.jl`),
```julia
P_hw = highest_weight_projector(Va ⊗ Vb, c)
@tensor HW_eq[ad a b; c] := Jpa[ad a; a'] * P_hw[a' b; c] + Jpb[ad b; b'] * P_hw[a b'; c]
CG_w = P_hw * rightnull(HW_eq; alg=SDD())'
```
so $C_"hw"$ is the *right null space* of @eq-hw (computed by SVD, `rightnull`), re-embedded through
`P_hw`. This is the reduced-basis form of the QSpace seed step @weichselbaum2012[App. B.1]: no
Casimirs are needed — a state killed by every raising operator is highest-weight by definition.

= Outer multiplicity and gauge <sec-mult>

The dimension of the null space of @eq-hw *is* the outer multiplicity $upright(N)_(a b)^c$: one HW
solution per independent coupling @alex2011. When $upright(N)_(a b)^c > 1$ (or the fused HW sector is
otherwise degenerate) the solutions span a space with no canonical basis — two resolutions are related
by an $O(m)$ (real) / $U(m)$ (complex) rotation, a pure *gauge choice* @pan1998. We fix it
deterministically with `gaugefix!` (column-reduced echelon form + positive-diagonal QR), the same
canonicalization the dense path uses:

#box-eq[
  `charge, bl = only(blocks(CG_w));` #h(6pt) `block(CGC, charge) .= gaugefix!(bl)`
]

The return value *must* be assigned: `gaugefix!` produces the orthonormal, gauge-fixed basis as a new
matrix and leaves its argument in a QR-mutated state, so discarding it leaves the HW block
non-orthonormal whenever it is degenerate (`blockdim > 1`). Because the gauge is a choice, degenerate
sectors agree with the literature only up to that $O(m)$ rotation @pan1998; the value is deterministic
but convention-dependent.

= Lower-weight descent <sec-descent>

The remaining weights follow by lowering. The CGC intertwines the lowering operators, giving the
recursion @alex2011[Eq. 40]

#box-eq[
  $ C_(w-1) med J^(-)_c = (J^(-)_a times.o bold(1) + bold(1) times.o J^(-)_b) med C_w, $ <eq-desc>
]

solved sector by sector for the lower block $C_(w-1)$ from the known higher block $C_w$. In code the
right-hand side is assembled from the current block and the coset lowering operators, and the (least
squares) solve of @eq-desc is the tensor right-division `rhs / eqs`:
```julia
Pw = highest_weight_projector(Vc, c)
while dim(Pw) != 0
    @tensor begin
        CGCw[a b; c n] := CGC[a b; c' n] * Pw[c'; c]
        rhs[a b n; ad c] := Jma[ad a; a'] * CGCw[a' b; c n] + Jmb[ad b; b'] * CGCw[a b'; c n]
        eqs[c; ad c'] := Jmc[ad c; c''] * Pw[c''; c']
    end
    @tensor CGC[a b; c n] += (rhs / eqs)[a b n; c]
    Pw = isometry(Vc, infimum(Vc, fuse(domain(Pw) ⊗ space(Jmc, 1)')))
end
```
The weight projector `Pw` starts at the HW sector of $V_c$ and steps down one coset charge at a time:
`fuse(domain(Pw) ⊗ space(Jmc, 1)')` lists the sectors reachable by one coset lowering, and `infimum`
intersects them with what actually occurs in $V_c$; the loop ends when no lower sector remains
($dim("Pw") = 0$). Because the coset lowering leg lowers the U(1) charge by $N$ and each step reaches
every SU($N-1$) content admissible at the next level, the descent visits *every* reduced sector of
$V_c$ — a fact the isometry check (@sec-code) certifies a posteriori. Summing both the $J^(-)_a$ and
$J^(-)_b$ contributions in `rhs` accounts for the several lowering paths into a degenerate weight
(inner multiplicity $> 1$ for $N >= 3$) @alex2011[Fig. 2].

= Connection to code and verification <sec-code>

#table(
  columns: 2, stroke: 0.4pt + gray, inset: 6pt, align: left,
  [*Object / step*], [*Code (`src/bootstrap.jl`)*],
  [full routine $a times.o b arrow.r c$], [`reduced_CGC`],
  [reduced spaces $V_a, V_b, V_c$], [`reduced_space`],
  [outer-multiplicity leg $V_n$], [`Vn`, sized by `Nsymbol(a,b,c)`],
  [HW sector projector], [`highest_weight_projector`],
  [HW null space (@eq-hw)], [`rightnull(HW_eq; alg=SDD())`],
  [multiplicity gauge fix], [`gaugefix!` (from `clebschgordan.jl`)],
  [coset raising / lowering $J^(plus.minus)$], [`reduced_raising_operators`, `reduced_lowering_operators`],
  [lowering recursion (@eq-desc)], [the `while dim(Pw) != 0` loop, `rhs / eqs`],
)

*Verification.* Correctness is established intrinsically, without a fragile dense-array comparison. A
map $C : V_a times.o V_b arrow.l V_c times.o V_n$ that (i) is an isometry and (ii)
intertwines all su($N$) generators *is* the CGC, unique up to the outer-multiplicity gauge. The
SU($N-1$)$times$U(1) generators are automatic from the `TensorMap`'s covariance, so only the *coset*
raising and lowering intertwiners need explicit checking. `test/bootstrap.jl` asserts, for
$N = 3, 4, 5$,
$
  norm(C^dagger C - bold(1)) < 10^(-10), wide
  norm((J_a times.o bold(1) + bold(1) times.o J_b) C - C med J_c) < 10^(-10)
$
for $J = J^(plus.minus)$. The isometry gate also certifies *descent completeness*: an unfilled sector
of $V_c$ would make $C^dagger C != bold(1)$ there. The SU(3) subblocks are additionally checked
against the tabulated de Swart / Kaeding isoscalar factors @deswart1963 @kaeding1995 (multiplicity-free
products, up to the overall-sign gauge). A direct comparison to the dense `CGC` is deliberately avoided:
`reduced_basistransform` reorders reduced *sector blocks* but not the states inside a sector, so it is
not a faithful dense bridge for irreps with larger SU($N-1$) sub-sectors.

= Phase 1 recap: coset operators <sec-phase1>

The lowering/raising operators consumed above are the SU($N$) *coset* ladder operators — the SU($N-1$)
fundamental / antifundamental carrying U(1) charge $minus.plus N$ — derived in closed form in the
Phase 1 note (`phase1-derivation.typ`). Because the branching is multiplicity-free, each is a single
scalar isoscalar factor per addable/removable box, written directly into a `TensorMap` subblock. The
raising element for adding a box in row $i$ is $r^"raise" = (-1)^i sqrt(abs(product_j (m_i - l_j) \/
product_(k eq.not i)(m_i - m_k)))$ in Molev shifted coordinates, and lowering is its rescaled
conjugate; see that note for the derivation and conventions.

= Sources <sec-sources>

#bibliography("references.bib", title: none, style: "american-physics-society")
