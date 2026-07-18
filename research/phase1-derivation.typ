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
  #text(15pt, weight: "bold")[Reduced SU($N$) ladder operators in an SU($N-1$)$times$U(1) basis]
  #v(2pt)
  #text(10pt)[Phase 1 derivation — `SUNRepresentations.jl`, branch `reduced`]
]

#v(4pt)

This note derives the SU($N$) *coset* raising and lowering operators as TensorKit `TensorMap`s over
SU($N-1$)$times$U(1) sectors, in closed form. These are the only generators needed to bootstrap
the reduced Clebsch–Gordan coefficients (`reduced_CGC`). Sources are collected in @sec-sources.

= Setup and conventions <sec-conv>

An SU($N$) irrep is a highest weight $lambda = (lambda_1 >= dots.c >= lambda_N)$ of integer Young-row
lengths (`SUNIrrep{N}`, `weight(a) = lambda`). Its basis states are Gelfand–Tsetlin (GT) patterns:
triangular arrays with top row $lambda^((N)) = lambda$ and rows $lambda^((k))$ obeying the
*betweenness* condition $lambda^((k))_i >= lambda^((k-1))_i >= lambda^((k))_(i+1)$.

Row $N-1$ of a pattern, $mu = (mu_1, dots, mu_(N-1))$, is the SU($N-1$) content under the
restriction SU($N$) $arrow.b$ SU($N-1$). This branching is *multiplicity-free* (each admissible
$mu$ occurs once), so each reduced sector below carries *degeneracy 1*.

#box-eq[
  *Reduced charge* $(nu, Y)$ of a sector (matches `reduced_charge`):
  $ nu = "normalize"(mu) quad (mu " with " mu_(N-1) " subtracted"), wide
    Y = N sum_(k=1)^(N-1) mu_k - (N-1) sum_(j=1)^N lambda_j . $
]

Given $(nu, Y)$ and $lambda$, the gl($N-1$) weight is recovered uniquely: $nu$ fixes $mu$ up to an
overall shift and $Y$ fixes $sum_k mu_k$ (`_reduced_weight`). Throughout we use the *shifted
coordinates*
$
  l_j = lambda_j - j + 1 quad (j = 1, dots, N), wide
  m_k = mu_k - k + 1 quad (k = 1, dots, N-1).
$

= CGCs and generators as `TensorMap`s <sec-tm>

By the Racah factorization lemma, an SU($N$) CGC factorizes along the chain SU($N$) $supset$
SU($N-1$)$times$U(1) into an SU($N-1$)$times$U(1) CGC times a *reduced coupling coefficient*
(isoscalar factor) that is independent of the SU($N-1$) internal labels @deswart1963 @molev2006.
The same statement for operators is the Wigner–Eckart theorem @weichselbaum2012.

In TensorKit @tensorkit2025 this is exactly how a symmetric map is stored: the coefficient for one
splitting/fusion-tree pair — the *subblock* `t[f₁, f₂]` — is the reduced matrix element, while the
SU($N-1$)$times$U(1) CGC is carried implicitly by the tree. With degeneracy 1, every subblock is a
single scalar.

The SU($N$) generators form a rank-1 tensor operator in the adjoint, which branches as
$
  "adj"("SU"(N)) arrow.b "adj"("SU"(N-1)) plus.o bold(1) plus.o "fund" plus.o "afund".
$ <eq-adj>
The `adj(SU(N-1))` block acts within each sector (supplied for free by the sector structure) and the
singlet is the diagonal U(1) charge; only the *coset* pieces — the SU($N-1$) fundamental /
antifundamental carrying U(1) charge $minus.plus N$ — must be built. These are the operators
$E_(i N)$ (raising) and $E_(N i)$ (lowering).

= Gelfand–Tsetlin matrix elements <sec-gt>

The action of the Chevalley generators on the GT basis is classical @molev2006[Thm. 2.3]
@alex2011[Eqs. 28–29]; with $l_(k i) = lambda^((k))_i - i + 1$,
$
  E_(k, k+1) xi_Lambda = - sum_(i=1)^k
    (product_(i') (l_(k i) - l_(k+1, i'))) / (product_(i' eq.not i) (l_(k i) - l_(k i')))
    xi_(Lambda + delta_(k i)),
$
and $E_(k+1, k)$ the analogous lowering form. Passing to the orthonormal basis
$zeta_Lambda = xi_Lambda \/ norm(xi_Lambda)$ (norm from @molev2006[Prop. 2.4]) symmetrizes these into
*signed square roots* — precisely what `creation` / `annihilation` compute in `gtpatterns.jl`.

For the branching step it is cleaner to use Molev's operators $z_(i N)$ (raising) and $z_(N i)$
(lowering), which map an SU($N-1$)-highest-weight vector of one sector directly to that of an
adjacent sector @molev2006[§2.1]. Their action is a closed product @molev2006[Lem. 2.13, Eq. 2.28]:
$
  z_(i N) xi_mu = - product_(j=1)^N (m_i - l_j) med xi_(mu + delta_i),
$ <eq-molev>
with $xi_(mu+delta_i) = 0$ unless $mu + delta_i$ is admissible. Because each SU($N-1$)-HW subspace
is one-dimensional, @eq-molev *is* the (unnormalized) isoscalar factor for adding a box in row $i$.

= Closed-form reduced matrix elements <sec-result>

Let a subblock connect $mu_"in"$ (domain) to $mu_"out"$ (codomain), where the coset step adds
(raising) or removes (lowering) a single box in row $i$: $mu_"out" = mu_"in" plus.minus delta_i$.

#box-eq[
  *Raising* ($mu_"out" = mu_"in" + delta_i$, evaluate $m$ at $mu_"in"$):
  $ r^("raise") = (-1)^i sqrt(abs(
      (product_(j=1)^N (m_i - l_j)) / (product_(k eq.not i) (m_i - m_k))
    )) . $
]

*Derivation.* @eq-molev gives the numerator $product_j (m_i - l_j)$. To turn $z_(i N)$ into the
physical tensor operator one divides by its h-factors $product_(k<i)(m_i - m_k)$ @molev2006[Eq. 2.17]
and orthonormalizes with the @molev2006[Prop. 2.4] norm ratio $norm(xi_(mu+delta_i)) \/ norm(xi_mu)$;
together these produce the level-$(N-1)$ denominator $product_(k eq.not i)(m_i - m_k)$ and the
square root. The sign $(-1)^i$ fixes the antifundamental tensor-operator convention (confirmed
against the raw generators, @sec-code).

#box-eq[
  *Lowering* ($mu_"out" = mu_"in" - delta_i$):
  $ r^("lower") = - sqrt(
      abs((product_(j) (m_i^"out" - l_j)) / (product_(k eq.not i) (m_i^"out" - m_k^"out")))
      dot (dim nu_"out") / (dim nu_"in")
    ), $
]
where $m^"out"$ is evaluated at $mu_"out"$. This is the Hermitian conjugate of the raising element
for $mu_"out" arrow.r mu_"in"$ (so the same product, now anchored at $mu_"out"$), rescaled by the
SU($N-1$) dimension ratio that relates the fundamental and antifundamental CGC normalizations.

*Conventions.* Raising $=$ `afund(SU(N-1)) ⊠ U1(-N)`, lowering $=$ `fund(SU(N-1)) ⊠ U1(+N)`; for
$N = 3$ the SU(2) factor is `SU2Irrep` (self-dual, so fund $=$ afund).

*Worked example (SU(3) fundamental).* $lambda = (1,0,0)$, so $l = (1, -1, -2)$. The single coset
transition is $mu_"in" = (0,0)$ #h(2pt) $[nu = bold(1), Y = -2]$ $arrow.r$ $mu_"out" = (1,0)$
#h(2pt) $[nu = bold(2), Y = +1]$, a box in row $i = 1$. Then $m_1 = 0$,
$
  r^("raise") = (-1)^1 sqrt(abs((0-1)(0+1)(0+2) / (0 - (-1)))) = -sqrt(2),
$
matching the golden value asserted in `test/reduced.jl`.

= Connection to code and verification <sec-code>

#table(
  columns: 2, stroke: 0.4pt + gray, inset: 6pt, align: left,
  [*Formula / object*], [*Code (`src/bootstrap.jl`)*],
  [reduced charge $(nu, Y)$, sector space], [`reduced_charge`, `reduced_space`],
  [recover $mu$ from $(nu, Y, lambda)$], [`_reduced_weight`],
  [$abs(product_j (m_i - l_j) \/ product_(k eq.not i)(m_i - m_k))$], [`_raise_mag2`],
  [raising / lowering operators], [`reduced_raising_operators`, `reduced_lowering_operators`],
  [coset sector `afund`/`fund` $⊠$ `U1`], [`_coset_antifund`, `_coset_fund`],
  [subblock fill $r$ over fusion trees], [`g[f₁, f₂] .= r` for `(f₁,f₂) in fusiontrees(g)`],
)

The operators never form a dense generator: each scalar $r$ is written directly into its subblock.
They are verified against the raw GT generators (built independently from `creation` /
`annihilation`) to machine precision for $N = 3, 4, 5$ in `test/reduced.jl`, which also checks that
every reduced sector has degeneracy 1, that subblocks are scalars, and the SU(3) $-sqrt(2)$ value.

= Sources <sec-sources>

#bibliography("references.bib", title: none, style: "american-physics-society")
