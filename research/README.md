# Research

Background literature and theory notes for the **SU(N-1)×U(1)-symmetric (TensorMap) CGC**
approach being developed on the `reduced` branch of `SUNRepresentations.jl`.

## Contents

| File | Purpose |
|------|---------|
| `references.md`   | Annotated reference list: links, one-paragraph summary, and relevance per source. |
| `references.bib`  | BibTeX entries for citing in code comments / docs. |
| `theory-notes.md` | Derivation: SU(N)↓SU(N-1)×U(1) branching, Racah factorization / isoscalar factors, and how these map onto TensorKit `TensorMap` CGCs. |
| `pdf-guide.md`    | Deep reader's guide to the PDFs: per-paper section→page maps, key equations, and how each part feeds implementation phases P1/P2/P3. Includes a "by phase" reading path and a cross-cutting conventions/gotchas list. |
| `implementation-plan.md` | The phased, citation-backed build plan synthesized from the analysis + literature: Phase 0 (cleanup) → 1 (general-N generators) → 2 (`reduced_CGC`) → 3 (F/R) → 4 (cache) → 5 (tests), with validation gates and conventions/risks. |
| `generator-construction.md` | Focused derivation for Phase 1: building the coset ladder operators explicitly, block-by-block, from closed-form reduced matrix elements (Molev z-operators / Lemma 2.13) instead of dense SVD-projection. |
| `phase1-derivation.typ` / `.pdf` | Phase 1 write-up: closed-form coset ladder operators. |
| `phase2-derivation.typ` / `.pdf` | Phase 2 write-up: `reduced_CGC` (HW null space + lowering descent) as a TensorMap of isoscalar factors. |
| `phase3-derivation.typ` / `.pdf` | Phase 3 write-up: `F`/`R` from a projector contraction on `reduced_CGC`; dense-gauge match for tower self-consistency; `rightnull` tolerance fix; iterative refinement of the descent. Records open follow-ups. |
| `phase3-gauge-handoff.md` | Superseded handoff note from the mid-Phase-3 investigation (kept for history; see the Phase 3 write-up for the resolved design). |
| `pdfs/`           | Local copies of the arXiv preprints (see below). |

## Fetching the PDFs

The PDFs are preprints, retained locally for convenience and subject to their original licenses.
Consider adding `research/pdfs/` to `.gitignore` if you do not want the binaries tracked.

Currently present in `pdfs/` (arXiv preprints for the freely-available items):

Core (§1–4 of `references.md`):
- `arXiv-1009.0437.pdf`  (Alex et al.)
- `arXiv-1202.5664.pdf`  (Weichselbaum, QSpace)
- `arXiv-1910.13736.pdf` (Weichselbaum, X-symbols)
- `arXiv-2508.10076.pdf` (TensorKit.jl)

Math / isoscalar-factor strand (§6):
- `arXiv-math_0211289.pdf`    (Molev, GT bases)
- `arXiv-nucl-th_9502037.pdf` (Kaeding, SU(3) isoscalar-factor tables — validation values)
- `arXiv-hep-th_9509167.pdf`  (Williams, SU(3) isoscalar-factor recursion)
- `arXiv-quant-ph_9704015.pdf`(Pan & Draayer, outer-multiplicity via complementary groups)

Journal-gated (no free preprint; links only in `references.md`): de Swart 1963, Hecht 1965,
Draayer & Akiyama 1973, Biedenharn & Louck 1968/1981, Louck 1970, Le Blanc & Rowe 1990,
SU3lib 2021, SU(4) CGC program.

To (re-)fetch the arXiv items from a normal shell:

```bash
cd research/pdfs
for id in 1009.0437 1202.5664 1910.13736 2508.10076 \
          math/0211289 nucl-th/9502037 hep-th/9509167 quant-ph/9704015; do
  curl -L -o "arXiv-$(echo "$id" | tr '/' '_').pdf" "https://arxiv.org/pdf/${id}"
done
```

Note: arXiv serves these at `https://arxiv.org/pdf/<id>` (a plain `curl -L` works). If `curl` fails
in a restricted/sandboxed shell, run it from an unrestricted terminal.

## Scope

This is a *living* folder. As the bootstrap approach develops, add:
- worked examples / derivations for specific N,
- benchmark notes comparing the reduced vs. dense CGC path,
- links to any upstream QSpace / TensorKit discussions.
