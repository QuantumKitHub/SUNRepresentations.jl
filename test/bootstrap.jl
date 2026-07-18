module BootstrapTests

using Test
using SUNRepresentations
using TensorKit
using LinearAlgebra
const SR = SUNRepresentations

# ===========================================================================
# Consolidated tests for the SU(N-1)×U(1) reduced-CGC machinery in
# `src/bootstrap.jl` (roadmap Phases 0-2). Organized in three layers:
#
#   A. Component units    — reduced_charge, reduced_space, _reduced_weight,
#                           reduced_basistransform, highest_weight_projector.
#   B. Coset generators   — reduced_raising/lowering_operators: block structure,
#                           dense agreement with the raw GT generators, the SU(3)
#                           golden value, and an su(N) algebra (commutator+Casimir) gate.
#   C. reduced_CGC        — isometry + coset-intertwiner gates (which together prove
#                           correctness up to the outer-multiplicity gauge), Nsymbol
#                           sizing, and an SU(3) isoscalar-factor cross-check vs the
#                           de Swart / Kaeding literature values.
#
# The file is self-contained (defines its own fixtures + independent oracles) so it
# can be run standalone with `julia --project=. -e 'include("test/bootstrap.jl")'`,
# bypassing the pre-existing TestExtras macro issue that aborts the full suite early.
# ===========================================================================

# Irreps used for the per-N sweeps. `IRREPS` is modest enough that the c ∈ a⊗b CGC
# sweep stays fast; `BIG_IRREPS` adds a few larger irreps for the generator-only
# checks (no fusion products, so cheap).
const IRREPS = Dict(
    3 => [SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(1, 1, 0), SUNIrrep{3}(2, 0, 0), SUNIrrep{3}(2, 1, 0)],
    4 => [SUNIrrep{4}(1, 0, 0, 0), SUNIrrep{4}(1, 1, 0, 0), SUNIrrep{4}(2, 0, 0, 0),
          SUNIrrep{4}(2, 1, 0, 0)],
    5 => [SUNIrrep{5}(1, 0, 0, 0, 0), SUNIrrep{5}(1, 1, 0, 0, 0), SUNIrrep{5}(2, 0, 0, 0, 0)],
)
const BIG_IRREPS = Dict(
    3 => [SUNIrrep{3}(3, 1, 0), SUNIrrep{3}(4, 2, 0)],
    4 => [SUNIrrep{4}(2, 1, 1, 0)],
    5 => SUNIrrep{5}[],
)
gen_irreps(N) = vcat(IRREPS[N], BIG_IRREPS[N])

# ---------------------------------------------------------------------------
# Independent reference implementations (ground truth), deliberately NOT sharing
# code with the constructors under test, so these tests guard any future
# reimplementation (e.g. the closed-form isoscalar-factor formula).
# ---------------------------------------------------------------------------

# (1) Frozen copy of the original SU(3)-only implementation -> regression anchor.
function su3_raising_ref(a::SUNIrrep{3}; tol=1.0e-12)
    p = SR.reduced_basistransform(a)
    Jp1, Jp2 = map(x -> x[p, p], SR.creation(a))
    Jp3 = SR.commutator(Jp1, Jp2)
    Va = SR.reduced_space(a)
    Vad = spacetype(Va)((1 // 2, -3) => 1)
    return TensorMap(convert(Array, stack((-Jp3, Jp2); dims=1)), Vad ⊗ Va ← Va; tol)
end
function su3_lowering_ref(a::SUNIrrep{3}; tol=1.0e-12)
    p = SR.reduced_basistransform(a)
    Jm1, Jm2 = map(x -> x[p, p], SR.annihilation(a))
    Jm3 = SR.commutator(Jm1, Jm2)
    Va = SR.reduced_space(a)
    Vad = spacetype(Va)((1 // 2, 3) => 1)
    return TensorMap(convert(Array, stack((-Jm2, Jm3); dims=1)), Vad ⊗ Va ← Va; tol)
end

# (2) Raw SU(N) coset generators from the GT ladder operators (creation/annihilation).
# This is the dense operator the reduced TensorMap must represent, for any N.
function raw_raising_ref(a::SUNIrrep{N}) where {N}
    p = SR.reduced_basistransform(a)
    Ep = map(x -> x[p, p], SR.creation(a))
    E = similar(Ep); E[N - 1] = Ep[N - 1]
    for i in (N - 2):-1:1
        E[i] = SR.commutator(Ep[i], E[i + 1])
    end
    return Float64.(convert(Array, stack([(-1)^k * E[k] for k in 1:(N - 1)]; dims=1)))
end
function raw_lowering_ref(a::SUNIrrep{N}) where {N}
    p = SR.reduced_basistransform(a)
    Em = map(x -> x[p, p], SR.annihilation(a))
    F = similar(Em); F[N - 1] = Em[N - 1]
    for i in (N - 2):-1:1
        F[i] = SR.commutator(F[i + 1], Em[i])
    end
    return Float64.(convert(Array, stack([-F[k] for k in (N - 1):-1:1]; dims=1)))
end

# ===========================================================================
# Layer A — component units
# ===========================================================================

@testset "reduced_charge partitions the multiplet (N=$N)" for N in 3:5
    for a in gen_irreps(N)
        counts = Dict{Any,Int}()
        for m in SR.basis(a)
            c = SR.reduced_charge(m)
            counts[c] = get(counts, c, 0) + 1
        end
        # every reduced charge is filled by exactly `dim(c)` GT states (degeneracy 1),
        # and the charges partition the whole multiplet.
        @test all(((c, n),) -> n == dim(c), counts)
        @test sum(values(counts)) == dim(a)
        # U(1) hypercharge is traceless over the multiplet (SU(N)-consistent normalization).
        @test sum(m -> SR.reduced_charge(m)[2].charge, SR.basis(a)) == 0
    end
end

@testset "reduced_space branching is multiplicity-free (N=$N)" for N in 3:5
    for a in gen_irreps(N)
        Va = SR.reduced_space(a)
        @test dim(Va) == dim(a)                       # total dimension preserved
        @test all(c -> dim(Va, c) == 1, sectors(Va))  # every reduced sector has degeneracy 1
    end
end

@testset "_reduced_weight inverts reduced_charge on sectors (N=$N)" for N in 3:5
    for a in gen_irreps(N)
        Va = SR.reduced_space(a)
        λ = weight(a)
        for c in sectors(Va)
            μ = SR._reduced_weight(c, λ)              # gl(N-1) weight of the sector
            # rebuild the SU(N-1) charge (row N-1 normalized) and the U(1) charge from μ,
            # and check they reproduce `c` (i.e. reduced_charge ∘ _reduced_weight == id).
            ν = SUNIrrep{N - 1}(μ)                   # weight ctor auto-normalises
            νr = N == 3 ? SU2Irrep(ν.I[1] // 2) : SUNIrrep{N - 1}(ν.I)
            Y = U1Irrep(N * sum(μ) - (N - 1) * sum(λ))
            @test (νr ⊠ Y) == c
        end
    end
end

@testset "reduced_basistransform is a sector-grouping permutation (N=$N)" for N in 3:5
    for a in gen_irreps(N)
        p = SR.reduced_basistransform(a)
        @test isperm(p)                               # valid permutation of 1:dim(a)
        @test length(p) == dim(a)
        # after permuting, states of one reduced charge form a contiguous block:
        charges = [SR.reduced_charge(m) for m in collect(SR.basis(a))[p]]
        seen = Set()
        contiguous = true
        for i in eachindex(charges)
            if i > 1 && charges[i] != charges[i - 1]
                charges[i] in seen && (contiguous = false)
                push!(seen, charges[i - 1])
            end
        end
        @test contiguous
    end
end

@testset "highest_weight_projector isolates the HW reduced sector (N=$N)" for N in 3:5
    for a in gen_irreps(N)
        Va = SR.reduced_space(a)
        P = SR.highest_weight_projector(Va, a)
        hwc = SR.reduced_charge(SR.highest_weight(a))
        @test P' * P ≈ id(domain(P))                  # isometry
        @test only(sectors(domain(P))) == (hwc,)      # onto exactly the HW sector (multiplicity 1)
        @test dim(domain(P)) == dim(Va, hwc) * dim(hwc)
    end
end

# ===========================================================================
# Layer B — coset generators
# ===========================================================================

@testset "coset operator subblocks are scalars (N=$N)" for N in 3:5
    for a in gen_irreps(N)
        for g in (SR.reduced_raising_operators(a), SR.reduced_lowering_operators(a))
            for (f₁, f₂) in fusiontrees(g)
                @test size(g[f₁, f₂]) == (1, 1, 1)
            end
        end
    end
end

@testset "SU(3) coset operators match the original implementation" begin
    for a in gen_irreps(3)
        @test convert(Array, SR.reduced_raising_operators(a)) ≈
              convert(Array, su3_raising_ref(a))
        @test convert(Array, SR.reduced_lowering_operators(a)) ≈
              convert(Array, su3_lowering_ref(a))
    end
end

# The load-bearing test for the closed form: the reduced TensorMap, expanded densely,
# must equal the raw SU(N) coset generator built from the GT ladder operators.
@testset "coset operators represent the raw SU(N) generators (N=$N)" for N in 3:5
    for a in gen_irreps(N)
        @test convert(Array, SR.reduced_raising_operators(a)) ≈ raw_raising_ref(a)
        @test convert(Array, SR.reduced_lowering_operators(a)) ≈ raw_lowering_ref(a)
    end
end

# Absolute golden value: SU(3) fundamental has a single coset transition (0,-2) -> (1/2,1)
# with reduced matrix element -sqrt(2). Anchors the normalization/sign convention outright.
@testset "SU(3) fundamental golden value" begin
    g = SR.reduced_raising_operators(SUNIrrep{3}(1, 0, 0))
    vals = [only(g[f₁, f₂]) for (f₁, f₂) in fusiontrees(g)]
    @test length(vals) == 1
    @test only(vals) ≈ -sqrt(2)
end

# --- su(N) algebra gate --------------------------------------------------
# The coset root vectors E_{iN} (raising) and E_{Ni} (lowering) generate all of su(N):
# the SU(N-1) block is E_{ij} = [E_{iN}, E_{Nj}] (i≠j), and the Cartan follows from
# E_{ii} - E_{NN} = [E_{iN}, E_{Ni}] together with Σ_i E_{ii} = |λ|·1. Recovering the
# operators from the densified reduced TensorMaps (with their tensor-operator sign/order
# conventions) and reconstructing the whole algebra lets us assert (i) the coset raisings
# commute, and (ii) the quadratic Casimir C₂ = Σ_{ij} E_{ij}E_{ji} equals the closed-form
# eigenvalue c_λ = Σ_k λ_k(λ_k - 2k + N + 1) times the identity. This validates the reduced
# operators at the Lie-algebra level, beyond the element-wise dense match above.
comm(x, y) = x * y - y * x

# Recover the standard gl(N) coset generators from the reduced operators.
# reduced_raising[i]  = (-1)^i E_{i,N};  reduced_lowering[m] = -E_{N,N-m}.
function _recover_coset(a::SUNIrrep{N}) where {N}
    Jp = convert(Array, SR.reduced_raising_operators(a))   # (N-1, d, d)
    Jm = convert(Array, SR.reduced_lowering_operators(a))  # (N-1, d, d)
    EiN = [(-1)^i * Jp[i, :, :] for i in 1:(N - 1)]
    ENi = [-Jm[N - i, :, :] for i in 1:(N - 1)]
    return EiN, ENi
end

function casimir_from_reduced(a::SUNIrrep{N}) where {N}
    EiN, ENi = _recover_coset(a)
    d = size(EiN[1], 1)
    E = Dict{Tuple{Int,Int},Matrix{Float64}}()
    for i in 1:(N - 1)
        E[(i, N)] = EiN[i]
        E[(N, i)] = ENi[i]
    end
    for i in 1:(N - 1), j in 1:(N - 1)
        i == j && continue
        E[(i, j)] = comm(E[(i, N)], E[(N, j)])          # SU(N-1) off-diagonal
    end
    diffs = [comm(E[(i, N)], E[(N, i)]) for i in 1:(N - 1)]  # E_ii - E_NN
    absλ = sum(weight(a))
    ENN = (absλ * Matrix{Float64}(I, d, d) - sum(diffs)) / N
    Ejj = Vector{Matrix{Float64}}(undef, N)
    Ejj[N] = ENN
    for i in 1:(N - 1)
        Ejj[i] = diffs[i] + ENN
    end
    C2 = zeros(Float64, d, d)
    for i in 1:N, j in 1:N
        C2 += (i == j) ? Ejj[i] * Ejj[i] : E[(i, j)] * E[(j, i)]
    end
    return C2
end

@testset "su(N) algebra gate: commutators + Casimir (N=$N)" for N in 3:5
    for a in gen_irreps(N)
        EiN, ENi = _recover_coset(a)
        d = size(EiN[1], 1)
        # coset raising operators commute among themselves (roots e_i - e_N add to non-roots).
        for i in 1:(N - 1), j in 1:(N - 1)
            @test norm(comm(EiN[i], EiN[j])) < 1e-10
            @test norm(comm(ENi[i], ENi[j])) < 1e-10
        end
        # quadratic Casimir is a scalar with the closed-form eigenvalue.
        λ = weight(a)
        c_λ = sum(λ[k] * (λ[k] - 2k + N + 1) for k in 1:N)
        C2 = casimir_from_reduced(a)
        @test norm(C2 - c_λ * I) < 1e-9
        @test C2[1, 1] ≈ c_λ
    end
end

# ===========================================================================
# Layer C — reduced_CGC
# ===========================================================================

# `reduced_CGC(a, b, c)` builds the SU(N) CGC as an SU(N-1)×U(1) TensorMap. We validate it with
# basis-independent (intrinsic) gates that together *prove* correctness up to the outer-multiplicity
# gauge — a map C : Va⊗Vb ← Vc⊗Vn that is an isometry AND intertwines all su(N) generators is the
# CGC (unique up to gauge). The generators split under SU(N)↓SU(N-1)×U(1) into the coset
# raising/lowering (built explicitly) plus the SU(N-1) adjoint + U(1) (supplied automatically by the
# TensorMap's SU(N-1)×U(1) covariance), so checking the coset intertwiners covers the nontrivial part.
# The isometry gate also certifies descent *completeness*: if any Vc sector went unfilled, C'C would
# not equal the identity there.
#
# We deliberately do NOT compare `convert(Array, reduced_CGC)` to the dense `CGC`: the only bridge
# available (`reduced_basistransform`) reorders the reduced sector blocks but not the states inside a
# sector, so it misaligns for irreps with larger SU(N-1) sub-sectors. The intrinsic gates below are
# both robust and complete; the literature cross-check anchors the actual isoscalar-factor values.

isometry_error(C) = norm(C' * C - id(domain(C)))

# ‖(J_a ⊗ 1 + 1 ⊗ J_b)·C − C·J_c‖, for a coset ladder operator family `J`.
# The denominator is floored at 1 so that singlet targets (where both sides are ≈ 0 — a singlet is
# annihilated by every generator) report an absolute residual instead of dividing noise by noise.
function intertwiner_error(C, Ja, Jb, Jc)
    @tensor L[ad x y; z n] := Ja[ad x; x'] * C[x' y; z n] + Jb[ad y; y'] * C[x y'; z n]
    @tensor R[ad x y; z n] := C[x y; z' n] * Jc[ad z'; z]
    return norm(L - R) / max(norm(L), norm(R), one(real(eltype(C))))
end

function check(a, b, c)
    C = SR.reduced_CGC(a, b, c)
    iso = isometry_error(C)
    rp = intertwiner_error(C, SR.reduced_raising_operators(a), SR.reduced_raising_operators(b),
                           SR.reduced_raising_operators(c))
    rm = intertwiner_error(C, SR.reduced_lowering_operators(a), SR.reduced_lowering_operators(b),
                           SR.reduced_lowering_operators(c))
    return iso, rp, rm
end

@testset "reduced_CGC isometry + intertwiners (N=$N)" for N in 3:5
    for a in IRREPS[N], b in IRREPS[N], c in a ⊗ b
        iso, rp, rm = check(a, b, c)
        @test iso < 1e-10                # isometry: C'C = 1 (also certifies descent completeness)
        @test rp < 1e-10                 # raising  intertwiner (independent of the construction)
        @test rm < 1e-10                 # lowering intertwiner
        # the outer-multiplicity leg is sized by Nsymbol.
        @test dim(domain(SR.reduced_CGC(a, b, c))[2]) == Nsymbol(a, b, c)
    end
end

# Explicit regression for the highest-weight-degeneracy cases (were badly wrong before the
# `gaugefix!` return-value fix): the target's HW reduced charge is degenerate in the fused space.
@testset "degenerate fused HW sector" begin
    for (a, b, c) in [(SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(1, 1, 0), SUNIrrep{3}(0, 0, 0)),  # 3⊗3̄→1
                      (SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(2, 1, 0), SUNIrrep{3}(1, 0, 0)),  # 3⊗8→3
                      (SUNIrrep{3}(2, 1, 0), SUNIrrep{3}(2, 1, 0), SUNIrrep{3}(2, 1, 0)),  # 8⊗8→8 (Nsym 2)
                      (SUNIrrep{4}(2, 1, 0, 0), SUNIrrep{4}(1, 1, 0, 0), SUNIrrep{4}(1, 0, 0, 0))]  # 20⁺⊗6→4
        iso, rp, rm = check(a, b, c)
        @test iso < 1e-10
        @test rp < 1e-10
        @test rm < 1e-10
    end
end

# --- SU(3) isoscalar-factor cross-check vs the literature -----------------
# The reduced_CGC subblocks C[f₁,f₂] *are* the SU(3)⊃SU(2)×U(1) isoscalar factors. For the
# multiplicity-free products below they are tabulated (de Swart, Rev. Mod. Phys. 35, 916 (1963);
# Kaeding, At. Data Nucl. Data Tables 61, 233 (1995)). Reference values keyed by the reduced charge
# channel (Ja,Ya)⊗(Jb,Yb)→(Jc,Yc). We compare up to the overall-sign gauge of each target block
# (an unavoidable de Swart/Condon-Shortley phase convention) by matching the value vector up to ±1.
_chan(f₁, f₂) = (f₁.uncoupled[1], f₁.uncoupled[2], f₂.uncoupled[1])

# helper: (2j, Y) shorthand -> SU2Irrep(j) ⊠ U1Irrep(Y)
_s(twoj, Y) = SU2Irrep(twoj // 2) ⊠ U1Irrep(Y)

# reference isoscalar factors (signed, de Swart convention)
const ISOSCALAR_REF = Dict(
    # 3 ⊗ 3 → 6 (symmetric): mixed terms same sign
    (SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(2, 0, 0)) => Dict(
        (_s(1, 1), _s(1, 1), _s(2, 2)) => 1.0,
        (_s(1, 1), _s(0, -2), _s(1, -1)) => 1 / sqrt(2),
        (_s(0, -2), _s(1, 1), _s(1, -1)) => 1 / sqrt(2),
        (_s(0, -2), _s(0, -2), _s(0, -4)) => 1.0),
    # 3 ⊗ 3 → 3̄ (antisymmetric): mixed terms opposite sign
    (SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(1, 1, 0)) => Dict(
        (_s(1, 1), _s(1, 1), _s(0, 2)) => 1.0,
        (_s(1, 1), _s(0, -2), _s(1, -1)) => -1 / sqrt(2),
        (_s(0, -2), _s(1, 1), _s(1, -1)) => 1 / sqrt(2)),
    # 3 ⊗ 3̄ → 1 (singlet)
    (SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(1, 1, 0), SUNIrrep{3}(0, 0, 0)) => Dict(
        (_s(1, 1), _s(1, -1), _s(0, 0)) => sqrt(2 / 3),
        (_s(0, -2), _s(0, 2), _s(0, 0)) => -1 / sqrt(3)),
    # 3 ⊗ 3̄ → 8 (Y=0 block is the classic singlet/octet rotation)
    (SUNIrrep{3}(1, 0, 0), SUNIrrep{3}(1, 1, 0), SUNIrrep{3}(2, 1, 0)) => Dict(
        (_s(1, 1), _s(1, -1), _s(0, 0)) => 1 / sqrt(3),
        (_s(0, -2), _s(0, 2), _s(0, 0)) => sqrt(2 / 3),
        (_s(1, 1), _s(1, -1), _s(2, 0)) => 1.0,
        (_s(1, 1), _s(0, 2), _s(1, 3)) => 1.0,
        (_s(0, -2), _s(1, -1), _s(1, -3)) => 1.0),
)

@testset "SU(3) isoscalar factors vs de Swart/Kaeding" begin
    for ((a, b, c), ref) in ISOSCALAR_REF
        C = SR.reduced_CGC(a, b, c)
        got = Dict(_chan(f₁, f₂) => only(C[f₁, f₂]) for (f₁, f₂) in fusiontrees(C))
        @test Set(keys(got)) == Set(keys(ref))       # same set of nonzero channels
        # assemble aligned value vectors and compare up to one overall sign per target block.
        ks = collect(keys(ref))
        vref = [ref[k] for k in ks]
        vgot = [got[k] for k in ks]
        @test isapprox(vgot, vref; atol=1e-10) || isapprox(vgot, -vref; atol=1e-10)
    end
end

# --- Compact gauge-fix: densification-free c-HW slice ------------------------
# `_dense_gauge_transform` no longer densifies the CGC to read the c-highest-weight slice; it builds
# that slice compactly with `_hw_slice` (isoscalar factors + a recursive one-level-down HW slice,
# bottoming out at a tiny SU(2) CGC). Keyed by GT-pattern identity, so the §4 intra-sector ordering
# is irrelevant. These gates were not possible before (the old comment above explains why the full
# reduced-vs-dense array comparison was avoided) — the HW slice sidesteps the intra-sector reorder.

# c-HW slice read from a *full densification* of `C`, keyed by (m₁,m₂) GT-pattern pair.
function dense_hw_slice(C, a, b, c)
    A = convert(Array, C)
    ba = collect(SR.basis(a))[SR.reduced_basistransform(a)]
    bb = collect(SR.basis(b))[SR.reduced_basistransform(b)]
    bc = collect(SR.basis(c))[SR.reduced_basistransform(c)]
    kc = findfirst(==(SR.highest_weight(c)), bc)
    d = Dict{Any,Vector{Float64}}()
    for i in axes(A, 1), j in axes(A, 2)
        v = collect(A[i, j, kc, :])
        norm(v) < SR.TOL_GAUGE && continue
        d[(ba[i], bb[j])] = v
    end
    return d
end

# assemble a slice Dict into the dense (m₁ outer, m₂ inner) enumeration order -> (#pairs × n).
function slicematrix(dict, a, b)
    rows = Vector{Float64}[]
    for m1 in SR.basis(a), m2 in SR.basis(b)
        haskey(dict, (m1, m2)) && push!(rows, dict[(m1, m2)])
    end
    return permutedims(reduce(hcat, rows))
end

# (1) refactor faithfulness: the compact slice equals the slice from a full densification.
@testset "compact _hw_slice == dense slice (N=$N)" for N in 3:4
    for a in IRREPS[N], b in IRREPS[N], c in a ⊗ b
        C = SR.reduced_CGC(a, b, c)
        h = SR._hw_slice(C, a, b, c)
        d = dense_hw_slice(C, a, b, c)
        @test Set(keys(h)) == Set(keys(d))
        @test maximum((norm(h[k] - d[k]) for k in keys(d)); init=0.0) < 1e-10
    end
end

# (2) gauge match vs the independent dense LAPACK solver (src/clebschgordan.jl). The gauge-fix must
# make reduced_CGC reproduce the dense convention: compare the c-HW slice keyed by GT patterns (the
# dense CGC is already in GT order; the HW state of c is the last basis index). Column spaces always
# coincide (any correct CGC); multiplicity-free channels additionally match up to the per-target sign.
@testset "reduced_CGC vs dense CGC on c-HW slice (N=$N)" for N in 3:4
    for a in IRREPS[N], b in IRREPS[N], c in a ⊗ b
        Mr = slicematrix(SR._hw_slice(SR.reduced_CGC(a, b, c), a, b, c), a, b)
        Cd = SR.CGC(Float64, a, b, c)
        ba = collect(SR.basis(a)); bb = collect(SR.basis(b)); kc = dim(c)
        dd = Dict{Any,Vector{Float64}}()
        for i in axes(Cd, 1), j in axes(Cd, 2)
            v = collect(Cd[i, j, kc, :])
            norm(v) < SR.TOL_GAUGE && continue
            dd[(ba[i], bb[j])] = v
        end
        Md = slicematrix(dd, a, b)
        @test size(Mr) == size(Md)
        @test norm(Mr * Mr' - Md * Md') < 1e-10            # column spaces coincide
        if Nsymbol(a, b, c) == 1
            @test isapprox(vec(Mr), vec(Md); atol=1e-10) || isapprox(vec(Mr), -vec(Md); atol=1e-10)
        end
    end
end

end
