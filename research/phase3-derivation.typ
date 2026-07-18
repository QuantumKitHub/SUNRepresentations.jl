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
  #text(15pt, weight: "bold")[$F$- and $R$-symbols from reduced SU($N$) Clebsch–Gordan `TensorMap`s]
  #v(2pt)
  #text(10pt)[Phase 3 derivation — `SUNRepresentations.jl`, branch `reduced`]
]

#v(4pt)

This note documents Phase 3: computing the SU($N$) $F$- and $R$-symbols by a *projector contraction*
on the compact SU($N-1$)$times$U(1) `reduced_CGC` `TensorMap`s (@sec-fr), with no densification of the
large SU($N$) CGC on the $F$/$R$ path. Three obstacles surfaced during implementation and each is
resolved here: the *tower must be self-consistent*, forcing `reduced_CGC` into the dense/GT gauge
(@sec-gauge); a *highest-weight null-space tolerance* bug that silently dropped whole channels
(@sec-null); and the *descent's accumulated recoupling noise*, cured by an intrinsic
iterative refinement (@sec-refine). Verification and open follow-ups close the note
(@sec-verify, @sec-followups). Builds on Phases 1–2 (`phase1-derivation.typ`, `phase2-derivation.typ`).

= $F$/$R$ from a projector contraction <sec-fr>

For a fusion category the $F$-symbol relates the two ways of coupling $a times.o b times.o c arrow.r d$,
and the $R$-symbol the two orders of $a times.o b arrow.r c$. In terms of CGCs (isometries
$C_(a b)^e : V_a times.o V_b arrow.l V_e times.o V_n$) the $F$-symbol is the overlap of the two trees
@weichselbaum2020[Eq. 16] @tensorkit2025:
$
  sum_(a b c e f) C_(a b)^e C_(e c)^d overline(C_(b c)^f) overline(C_(a f)^d)
    = dim(d) dot.c F^(a b c)_d [(e, mu_1, mu_2), (f, mu_3, mu_4)],
$
tracing the shared $d$ line ($C^dagger C = bold(1)$ gives $"Tree"_R^dagger compose "Tree"_L = F times.o bold(1)_d$
by Schur, hence the factor $dim(d)$). All four outer-multiplicity (OM) legs stay open, so for SU($N >= 3$)
$F$ is a genuine rank-4 tensor over $(mu_1 mu_2; mu_3 mu_4)$ @weichselbaum2020[Eq. 17]. In code
(`src/sector.jl`), with the reduced CGCs as the compact operands:

#box-eq[
```julia
A = reduced_CGC(a,b,e); B = reduced_CGC(e,c,d); C = reduced_CGC(b,c,f); D = reduced_CGC(a,f,d)
@tensor F[ne nd1; nf nd2] := A[α β; ε ne]*B[ε γ; δ nd1]*conj(C[β γ; φ nf])*conj(D[α φ; δ nd2])
return (convert(Array, F) ./ dim(d))::Array{Float64,4}
```
]

The `::Array{Float64,4}` type-assertion is mandatory: `reduced_CGC` is cached as `Any`, so without it
`sectorscalartype(I)` infers `Any` (via `Core.Compiler.return_type`) and TensorKit calls
`zeros(Any, …)` when building product-sector tensors — a `MethodError`. A `convert` does *not* propagate
through inference; the `::` does. The $R$-symbol is the analogous single-line contraction over $c$,
$sum_(a b) C_(a b)^c overline(C_(b a)^c) = dim(c) R^(a b)_c$. Only the small rank-2/4 result is ever
densified — never a CGC.

= Tower self-consistency forces the dense gauge <sec-gauge>

Building `reduced_CGC(SU(N))` consumes SU($N-1$) $F$-symbols (through the `@tensor` fusion-tree
recouplings), and the Phase-1 coset ladder operators are written in the Gelfand–Tsetlin (GT) gauge
@molev2006. So if $F$/$R$ are taken from `reduced_CGC` and fed back up the tower, `reduced_CGC(SU(N-1))`
*must* reproduce the dense/GT gauge, or the level-$N$ build stops intertwining
(measured failure with a mismatched gauge: $norm((J^(+)_a times.o bold(1) + bold(1) times.o J^(+)_b) C - C J^(+)_c) approx 2$,
$F$-unitarity broken at $N = 4$).

The gauge freedom is exactly the outer-multiplicity basis: an overall sign per multiplicity-free
channel, an $O(m)$ rotation per $m$-fold channel @pan1998. The dense `CGC` fixes it by
`gaugefix! = qrpos! ∘ cref!` applied to the HW null space with rows in `basis(a)⊗basis(b)`
*enumeration order* ($m_1$ outer, $m_2$ inner) @alex2011. `reduced_CGC` gauge-fixes in the coarser
reduced-sector order, giving a *different* per-channel sign / $O(m)$. We reconcile them with
`_dense_gauge_transform` (`src/bootstrap.jl`): read off the couplings to the SU($N$)-highest-weight
state of $c$ (the "c-HW slice", which keeps the OM leg separate), re-apply the dense `gaugefix!` in the
dense enumeration order, and read off the orthogonal change of OM basis $G$. Because the descent is
linear in the OM leg, one $G$ re-gauges every weight at once:

#box-eq[
  `G = _dense_gauge_transform(CGC, a, b, c);` #h(4pt)
  $C arrow.r C dot.c G med (G : V_n arrow.l V_n, med G^top G = bold(1)).$
]

Rows are matched by GT-pattern *identity*, not position, so the reduced-basis internal-ordering
divergence (`reduced_basistransform` orders sector blocks but not states within a sector) cannot corrupt
the match. With $G$ applied, `convert(Array, reduced_CGC)` equals the dense `CGC` to $tilde 10^(-15)$
including sign (multiplicity-free) / up to $O(m)$ (degenerate), and the projector $F$/$R$ equal the dense
reference-state $F$/$R$.

= A silent null-space drop at large irreps <sec-null>

`reduced_CGC`'s HW seed is the right null space of the coset-raising equation @eq-hwnull:
#box-eq[
  $ (J^(+)_a times.o bold(1) + bold(1) times.o J^(+)_b) med C_"hw" = 0 $ <eq-hwnull>
]
computed by `rightnull(HW_eq; alg = SDD())`. For larger irreps the coset-raising `HW_eq` accumulates
enough round-off that its *true* null singular value drifts up to $tilde 10^(-11)$ — above `rightnull`'s
default cut-off, which then returns an *empty* null space and drops the channel entirely (e.g.
`15⊗15→1` at $N=4$: `only(blocks(CG_w))` throws on an empty result). Genuine couplings have singular
values $O(1)$, so a relative tolerance separates them cleanly:

#box-eq[
  `CG_w = P_hw * rightnull(HW_eq; alg=SDD(), rtol=1.0e-9)'`
]

This is invisible in Phases 0–2 because `reduced_CGC` was tested only on the small irreps
${bold(4), bold(6), bold(10), bold(20)}$ (bootstrap gates), whereas the category gates request the
larger fusion products (`15`, `20⁺`, `64`, …) as intermediates.

= Descent noise and iterative refinement <sec-refine>

With the gauge matched and the null space recovered, the $F$/$R$ *values* are correct but the reduced
path is $tilde 100 times$ less accurate than the dense path at $N = 4$: `reduced_CGC` isometry error
grows $9 times 10^(-15)$ ($N=3$) $arrow.r 10^(-12)$ ($N=4$), so
$abs(F_"red" - F_"dense") dot.c dim(d) tilde 10^(-11)$ — over the strict category gate
($1000 dot.c epsilon.alt approx 2.2 times 10^(-13)$, effective bound $"rtol" dot.c norm(f_2)$).

*Diagnosis.* The per-shell descent solves `rhs / eqs` are *perfectly conditioned* — every `eqs` block
is a scalar multiple of an isometry, $kappa = 1$ at every level, $N=3$ and $N=4$ alike — so a
"better linear solver" (QR/SVD vs. LU) buys nothing. The error is instead *recoupling noise*: at $N=4$
every `@tensor` in the descent recouples through SU($3$) fusion trees, injecting the SU($3$)
$F$-symbols' $tilde 10^(-14)$ error, which the (exact) solve then faithfully accumulates down the ladder.

*Cure.* The exact CGC is the unique isometry annihilated by the coset-intertwiner defect
$Phi(C) = plus.o.big_(J in {J^plus, J^minus}) [(J_a times.o bold(1) + bold(1) times.o J_b) C - C J_c]$,
and the coset operators are *exact* ($sqrt(dot)$ of rationals). So we project the descent output onto the
true null space of $Phi$ and re-orthonormalize (polar):

#box-eq[
  $ C_"refined" = "polar"(cal(P)_(ker Phi) med C_"descent"), wide
    dim ker Phi = (upright(N)_(a b)^c)^2. $
]

Crucially this is done *compactly*: $Phi$ is assembled as a small matrix over the isoscalar-factor block
coefficients (`_cgc_coeffs`, dimension 4–38 in tested channels), not the full densification, its null
space taken by SVD (`_refine_reduced_CGC` in `src/bootstrap.jl`), and the polar factor via `tsvd`
(`U*Vᴴ`, real — `inv(sqrt(C'C))` would promote to complex). The projection restores
$tilde 10^(-14)$ / machine accuracy (isometry $10^(-12) arrow.r 10^(-14)$, direction error
$10^(-13) arrow.r 5 times 10^(-15)$) at $tilde 1"s"$ per CGC, and — because the tower now feeds on a
clean `reduced_CGC` — stops the error compounding upward.

= Verification <sec-verify>

#table(
  columns: 2, stroke: 0.4pt + gray, inset: 6pt, align: left,
  [*Gate*], [*Result*],
  [`reduced_CGC` vs dense `CGC` (GT-pattern bridge)], [$approx 10^(-15)$ incl. sign / up to $O(m)$, $N=3,4$],
  [$F$/$R$ vs dense reference-state oracle], [$approx 10^(-15)$ ($N=3$), $approx 4 times 10^(-12)$ ($N=5$)],
  [category gates: fusiontensor$↔$$F$/$R$, $F$-unitarity, pentagon, hexagon], [*all green, 0 failures, $N=3$ and $N=4$*],
  [bootstrap intrinsic gates (isometry + coset intertwiners)], [green, $N=3,4,5$ ($<10^(-10)$)],
  [type inference], [`return_type(Fsymbol, NTuple{6,I}) == Array{Float64,4}`, `sectorscalartype == Float64`],
)

The category gate at $N = 4$ (708 fusiontensor$↔$$F$/$R$ + 400 $F$-unitarity + 625 pentagon +
125 hexagon) passes with a *fully compact construction* — no dense CGCs, no tower-feedback workaround.
$N=3$ likewise. The full $N=5$ category gate was not run (its pentagon alone is $tilde$hours); the
$N=5$ $F$-vs-oracle check ($4 times 10^(-12)$) and the intrinsic gates stand in for it.

= Follow-ups <sec-followups>

+ *Compact gauge-fix* — *DONE (2026-07-18).* `_dense_gauge_transform` no longer densifies. The c-HW
  slice $M$ is now built by a recursive `_hw_slice(CGC, a, b, c)`: isoscalar factors read from the
  passed CGC's fusion-tree subblocks are combined (Racah) with the SU($N-1$) coupling to the HW state
  of $nu_"hw"$ — itself a one-level-down HW slice — bottoming out at a tiny SU($2$) CGC. Full GT
  patterns $m_1$ are rebuilt from $(s_a, m_a)$ by the uniform shift $mu = nu + "shift"$, and couplings
  are keyed by GT-pattern identity, so the §4 intra-sector ordering divergence never enters. Because
  $G = M \\ "gaugefix!"(M)$ depends only on $M$ (and, via `cref!`, only on its pivot rows), reusing the
  existing `gaugefix!`/`M \\ Q` verbatim gives a $G$ byte-identical to the old dense path — so
  `reduced_CGC` and the $F$/$R$ are unchanged. For the common multiplicity-free channels this makes the
  gauge a product of analytically-known signs (the pivot's sign) read from compact data; degenerate
  channels reduce to a tiny $n times n$ QR. Verified: compact slice $=$ dense slice to $tilde 10^(-16)$,
  and `reduced_CGC` matches the independent dense LAPACK solver on the c-HW slice (N=3,4). `src/bootstrap.jl`
  now has zero `convert(Array, dot)`; the only remaining densifications ($F$/$R$) act on the rank-2/4
  output, never a CGC.

+ *Reference-state $F$/$R$.* The residual $tilde 4 times 10^(-12)$ on $F$ is the `/dim(d)` full-trace
  contraction's own float accumulation, and it caps the tower (the $N=5$ isometry $tilde 10^(-13)$
  reflects SU($4$) $F$-symbols feeding the SU($5$) refinement). A single-reference-state contraction
  (no $d$ trace) would feed cleaner $F$-symbols upward and matter for $N >= 6$ / machine-precision
  through the tower — it needs single-internal-state extraction of $c$/$d$ in the reduced rep, which the
  block structure does not expose directly.

+ *Full $N=5$ category gate.* Confirm pentagon/hexagon at $N=5$ (and the `SUNIrrep{3} ⊠ SUNIrrep{3}`
  product sector) once runtime allows; possibly gate a reduced `smallset` for speed.

+ *Phases 4–5.* Restore the disk cache for the `TensorMap` path (`_get_CGC`, `src/caching.jl`); wire the
  category gates into `runtests.jl` for the reduced $F$/$R$ path; add the README note on the reduced
  approach. The RAM memoization (`REDUCED_CGC_CACHE`, `REDUCED_COSET_CACHE`) already landed.

+ *Refinement scaling.* `_refine_reduced_CGC` builds $Phi$ column-by-column ($tilde$`cdim` `@tensor`
  calls). Fine for tested channels; for very large irreps consider a matrix-free null-space iteration.

= Sources <sec-sources>

#bibliography("references.bib", title: none, style: "american-physics-society")
