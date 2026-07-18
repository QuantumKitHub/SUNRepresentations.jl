#=
Functionality for writing SU(N) CGCs in terms of SU(N-1) x U(1)
=#

# Compute SU(N-1) by discarding the first row and renormalizing (make last entry 0)
# Compute U(1) by how far down the second row is from the first, but shift such that
# total weight in the multiplet is 0
function reduced_charge(g::GTPattern{N}) where {N}
    # `SUNIrrep{N - 1}` weight constructor auto-normalises (last entry 0), replacing the old
    # `_normalize` helper that main removed.
    I = SUNIrrep{N - 1}(ntuple(i -> g[i, N - 1], N - 1))
    Y = U1Irrep(N * rowsum(g, N - 1) - (N - 1) * rowsum(g, N))
    if N == 3
        I = SU2Irrep(I.I[1] // 2)
    end
    return I ⊠ Y
end

function reduced_space(a::SUNIrrep{N}) where {N}
    # Sector type must be concrete: on main `SUNIrrep{N-1}` alone is the abstract `SUNIrrep{N-1,M}`,
    # so spell out the second parameter `N - 2`.
    I = (N == 3 ? SU2Irrep : SUNIrrep{N - 1, N - 2}) ⊠ U1Irrep
    dict = Dict{I, Int}()
    for m in basis(a)
        c = reduced_charge(m)
        dict[c] = get(dict, c, 0) + 1
    end

    return Vect[I](c => d ÷ dim(c) for (c, d) in dict)
end

function reduced_basistransform(a::SUNIrrep{N}) where {N}
    return sortperm(collect(basis(a)); by = x -> (reduced_charge(x), U1Irrep(rowsum(x, N - 2))))
end

adjoint_irrep(::SUNIrrep{N}) where {N} = SUNIrrep{N}(2, 1, ntuple(Returns(0), N - 2)...)
adjoint_irrep(::SUNIrrep{2}) = SUNIrrep{2}(2, 0)
fundamental_irrep(::SUNIrrep{N}) where {N} = SUNIrrep{N}(1, ntuple(Returns(0), N - 1)...)

commutator(x, y) = x * y - y * x

function reduced_generators(a::SUNIrrep{3}; tol = 1.0e-12)
    p = reduced_basistransform(a)
    Jp1, Jp2 = map(x -> x[p, p], creation(a))
    Jm1, Jm2 = map(x -> x[p, p], annihilation(a))
    Jz1 = commutator(Jp1, Jm1) ./ sqrt(2)
    Jz2 = commutator(Jp2, Jm2) ./ sqrt(2)
    Jp3 = commutator(Jp1, Jp2)
    Jm3 = commutator(Jm1, Jm2)

    Va = reduced_space(a)
    ad = adjoint_irrep(a)
    Vad = reduced_space(ad)

    adjoint_generators = stack((Jp1, Jz1, -Jm1); dims = 1)
    raising_generators = stack((-Jp3, Jp2); dims = 1)
    lowering_generators = stack((Jm2, -Jm3); dims = 1)
    singlet_generator = reshape(Jz1 + 2Jz2, 1, dim(Va), dim(Va))

    g_array = cat(singlet_generator, adjoint_generators, lowering_generators, raising_generators; dims = 1)
    # TK 0.17's `TensorMap(data, V; tol)` requires a dense array; `cat`/`stack` of `SparseArray`s
    # yields a `SparseArray`, so densify.
    g = TensorMap(convert(Array, g_array), Vad ⊗ Va ← Va; tol)
    return g
end

# Coset ladder operators in the SU(N-1) x U(1) reduced basis, built explicitly block by block
# for arbitrary N from a closed-form reduced matrix element (isoscalar factor) — no dense
# generator is ever formed. Because the branching SU(N) -> SU(N-1) is multiplicity-free, each
# reduced sector has degeneracy 1, so every fusion-tree subblock is a single scalar. The SU(N-1)
# x U(1) CGC is carried implicitly by the fusion tree. See research/generator-construction.md.
# Conventions: raising = antifund(SU(N-1)) ⊠ U1(-N); lowering = fund(SU(N-1)) ⊠ U1(+N).
# The SU(N-1) factor uses SU2Irrep for N=3 (matching `reduced_charge`) and SUNIrrep{N-1} else.
_coset_fund(::Val{3}) = SU2Irrep(1 // 2)
_coset_antifund(::Val{3}) = SU2Irrep(1 // 2)
_coset_fund(::Val{N}) where {N} = SUNIrrep{N - 1}(ntuple(i -> i == 1 ? 1 : 0, N - 1))
_coset_antifund(::Val{N}) where {N} = SUNIrrep{N - 1}(ntuple(i -> i < N - 1 ? 1 : 0, N - 1))

# Reconstruct the (unnormalized) gl(N-1) weight μ (length N-1) of a reduced sector `c = ν ⊠ Y`,
# given the gl(N) highest weight λ. ν fixes μ up to an overall shift; Y = N·Σμ − (N-1)·Σλ fixes it.
function _reduced_weight(c, λ::NTuple{N,Int}) where {N}
    ν, Y = c[1], c[2].charge
    νw = ν isa SU2Irrep ? (Int(2 * ν.j), 0) : weight(ν)   # normalized, last entry 0
    shift = ((Y + (N - 1) * sum(λ)) ÷ N - sum(νw)) ÷ (N - 1)
    return ntuple(k -> νw[k] + shift, N - 1)
end

# Squared magnitude of the coset raising reduced matrix element for adding a box in row `i` to
# the gl(N-1) weight μ (with gl(N) highest weight λ). In shifted coordinates m_k = μ_k - k + 1,
# l_j = λ_j - j + 1:  |num/den|, num = ∏_j (m_i - l_j), den = ∏_{k≠i}(m_i - m_k).
function _raise_mag2(μ::NTuple{M,Int}, λ::NTuple{N,Int}, i::Int) where {M,N}
    m = ntuple(k -> μ[k] - k + 1, M)
    num = prod(m[i] - (λ[j] - j + 1) for j in 1:N)
    den = prod(m[i] - m[k] for k in 1:M if k != i)
    return abs(num // den)
end

function reduced_raising_operators(a::SUNIrrep{N}) where {N}
    λ = weight(a)
    Va = reduced_space(a)
    Vad = spacetype(Va)(_coset_antifund(Val(N)) ⊠ U1Irrep(-N) => 1)
    g = zeros(Float64, Vad ⊗ Va ← Va)
    for (f₁, f₂) in fusiontrees(g)
        μin = _reduced_weight(f₂.coupled, λ)
        μout = _reduced_weight(f₁.uncoupled[2], λ)
        i = findfirst(k -> μout[k] - μin[k] == 1, 1:(N - 1))   # added box
        (g[f₁, f₂]) .= (-1)^i * sqrt(_raise_mag2(μin, λ, i))
    end
    return g
end

function reduced_lowering_operators(a::SUNIrrep{N}) where {N}
    λ = weight(a)
    Va = reduced_space(a)
    Vad = spacetype(Va)(_coset_fund(Val(N)) ⊠ U1Irrep(N) => 1)
    g = zeros(Float64, Vad ⊗ Va ← Va)
    for (f₁, f₂) in fusiontrees(g)
        μin = _reduced_weight(f₂.coupled, λ)
        μout = _reduced_weight(f₁.uncoupled[2], λ)
        i = findfirst(k -> μin[k] - μout[k] == 1, 1:(N - 1))   # removed box
        νin, νout = f₂.coupled[1], f₁.uncoupled[2][1]
        # Hermitian conjugate of the raising element from μout -> μin, rescaled by the SU(N-1)
        # dimension ratio that relates the fund and antifund CGC normalizations.
        (g[f₁, f₂]) .= -sqrt(_raise_mag2(μout, λ, i) * (dim(νout) // dim(νin)))
    end
    return g
end

function weight_space(V, a::SUNIrrep, w::Integer)
    I = (a.N == 3 ? SU2Irrep : SUNIrrep{a.N - 1, a.N - 2}) ⊠ U1Irrep
    set = Set{I}()
    hw = weight(highest_weight(a))
    for m in basis(a)
        c = reduced_charge(m)
        if sum(abs.(weight(m) .- hw)) == 2w
            push!(set, c)
        end
    end
    return Vect[I](c => blockdim(V, c) for c in set)
end
weight_projector(V, c, w) = isometry(V, weight_space(V, c, w))

function highest_weight_space(V, c)
    c = reduced_charge(highest_weight(c))
    return spacetype(V)(c => blockdim(V, c))
end
highest_weight_projector(V, c) = isometry(V, highest_weight_space(V, c))

function heightmap(a::SUNIrrep{N}) where {N}
    ba = Set(basis(a))
    result = Vector{eltype(ba)}[]
    hw = highest_weight(a)
    pop!(ba, hw)
    push!(result, [hw])
   
    while !isempty(ba)
        next = Vector{eltype(ba)}()
        for m in last(result)
            for l in 1:(N-1), k in 1:l
                m′ = Base.setindex(m, m[k, l] - 1, k, l)
                if m′ in ba
                    pop!(ba, m′)
                    push!(next, m′)
                end
            end
        end
        push!(result, next)
    end

    return result
end


# Bare SU(N-1) irrep of a reduced-sector factor. The SU(N-1) part is stored as `SUNIrrep{N-1}`,
# except the N=3 base case which uses TensorKit's `SU2Irrep` (matching `reduced_charge`).
_bare_sun(ν::SUNIrrep) = ν
_bare_sun(ν::SU2Irrep) = SUNIrrep{2}(Int(2 * ν.j), 0)

# Compact c-highest-weight slice of a reduced CGC: the couplings ⟨c-HW | m₁,m₂⟩ₙ for every fused
# dense pair (m₁,m₂) (full GT patterns) that reaches the SU(N)-highest-weight state of c, keyed by
# the (m₁,m₂) GT-pattern pair and valued by the length-`Nsymbol(a,b,c)` outer-multiplicity vector.
#
# This is built WITHOUT densifying any large CGC (the project's core goal). By Racah factorization,
#   ⟨c-HW | m₁,m₂⟩ₙ = Σ_κ  isoₙ[(sa,sb,κ)→ν_hw] · F_κ[(sa,ma),(sb,mb)]
# where `iso` are the compact isoscalar-factor block entries of the passed `CGC` at the HW reduced
# sector ν_hw⊠Y, and `F_κ` is the SU(N-1)×U(1) CG coupling to the HW state of ν_hw — itself a
# one-level-down HW slice. So `M` is assembled recursively over `reduced_CGC` of lower irreps and
# bottoms out at a tiny SU(2) CGC. Full patterns m₁ are rebuilt from (sa, ma) by uniform GT shift.
# Keying by GT-pattern identity sidesteps the §4 intra-sector ordering divergence of `convert`.
function _hw_slice(a::SUNIrrep{2}, b::SUNIrrep{2}, c::SUNIrrep{2})
    # base case: dense SU(2) CGC (O(1), never a large irrep); HW state of c is the last basis index.
    C = CGC(Float64, a, b, c)
    ba = collect(basis(a))
    bb = collect(basis(b))
    kc = dim(c)
    result = Dict{Tuple{GTPattern{2}, GTPattern{2}}, Vector{Float64}}()
    for i in axes(C, 1), j in axes(C, 2)
        v = collect(C[i, j, kc, :])
        norm(v) < TOL_GAUGE && continue
        result[(ba[i], bb[j])] = v
    end
    return result
end
function _hw_slice(a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where {N}
    return _hw_slice(reduced_CGC(a, b, c), a, b, c)
end
function _hw_slice(CGC, a::SUNIrrep{N}, b::SUNIrrep{N}, c::SUNIrrep{N}) where {N}
    λa = weight(a)
    λb = weight(b)
    n = Nsymbol(a, b, c)
    s_hw = reduced_charge(highest_weight(c))          # ν_hw ⊠ Y
    νhw = _bare_sun(s_hw[1])
    result = Dict{Tuple{GTPattern{N}, GTPattern{N}}, Vector{Float64}}()
    subcache = Dict{Tuple{SUNIrrep{N - 1, N - 2}, SUNIrrep{N - 1, N - 2}}, Any}()
    for (f₁, f₂) in fusiontrees(CGC)
        f₂.coupled == s_hw || continue
        sa, sb = f₁.uncoupled                         # reduced sectors νa⊠Ya, νb⊠Yb on the a,b legs
        κ = f₁.vertices[1]                            # SU(N-1)×U(1) inner fusion multiplicity
        iso = vec(CGC[f₁, f₂][1, 1, 1, :])            # isoscalar factors, one per outer-mult index n
        νa = _bare_sun(sa[1])
        νb = _bare_sun(sb[1])
        sub = get!(subcache, (νa, νb)) do
            _hw_slice(νa, νb, νhw)                     # recurse: SU(N-1) HW-slice couplings
        end
        shift_a = _reduced_weight(sa, λa)[1] - weight(νa)[1]   # uniform GT shift ν → μ
        shift_b = _reduced_weight(sb, λb)[1] - weight(νb)[1]
        for ((ma, mb), fvec) in sub
            m1 = GTPattern{N}((λa..., (ma.data .+ shift_a)...))
            m2 = GTPattern{N}((λb..., (mb.data .+ shift_b)...))
            r = get!(() -> zeros(Float64, n), result, (m1, m2))
            r .+= iso .* fvec[κ]
        end
    end
    for (k, v) in collect(result)
        norm(v) < TOL_GAUGE && delete!(result, k)
    end
    return result
end

# The dense/GT `CGC` fixes the outer-multiplicity gauge by `gaugefix! = qrpos!∘cref!` applied to
# the highest-weight null space, whose rows are the fused (m₁,m₂) states at the highest weight of c
# in `basis(a)⊗basis(b)` enumeration order (m₁ outer, m₂ inner). `reduced_CGC`'s descent fixes the
# same gauge but in the coarser reduced-sector row basis, giving a different per-channel sign / SO(m).
# This returns the orthogonal transform `G` on the outer-multiplicity leg mapping the reduced CGC
# onto the dense convention: build the c-HW slice `M` (couplings ⟨c-HW | m₁ m₂⟩ₙ), re-apply the dense
# `gaugefix!` in the dense enumeration order, and read off the change of multiplicity basis `G = M\Q`.
# Because `gaugefix!` depends only on the column space and the row order, applying it to the couplings
# re-expressed in the dense order *is* the dense convention. Rows are keyed by GT-pattern identity, so
# the §4 reduced-basis intra-sector ordering divergence cannot corrupt it.
#
# `M` is built compactly by `_hw_slice` (isoscalar factors + a recursive one-level-down HW slice,
# bottoming out at a tiny SU(2) CGC) — no large CGC is ever densified, and `M` is only `#pairs × n`.
function _dense_gauge_transform(CGC, a::I, b::I, c::I) where {I <: SUNIrrep}
    n = dim(domain(CGC)[2])
    n == 0 && return zeros(Float64, 0, 0)
    coeff = _hw_slice(CGC, a, b, c)                # c-HW slice keyed by (m₁,m₂); no densification
    # assemble in the dense enumeration order (m₁ outer over basis(a), m₂ inner over basis(b)).
    rows = Vector{Float64}[]
    for m1 in basis(a), m2 in basis(b)
        haskey(coeff, (m1, m2)) && push!(rows, coeff[(m1, m2)])
    end
    M = permutedims(reduce(hcat, rows))            # (#pairs × n), orthonormal columns
    Q = gaugefix!(copy(M))
    return M \ Q
end

# Flatten / restore a reduced-CGC (or defect) TensorMap over its block coefficients — the compact
# isoscalar-factor vector, far smaller than a full densification.
_cgc_coeffs(t) = reduce(vcat, (vec(block(t, s)) for s in blocksectors(t)); init = Float64[])
function _cgc_setcoeffs!(t, v)
    i = 1
    for s in blocksectors(t)
        b = block(t, s)
        n = length(b)
        vec(b) .= view(v, i:(i + n - 1))
        i += n
    end
    return t
end

# Coset-intertwiner defect Φ(C) = ⊕_{gen} [(Ja⊗1 + 1⊗Jb)·C − C·Jc], as a flat coefficient vector.
# gens = ((Jp_a,Jp_b,Jp_c), (Jm_a,Jm_b,Jm_c)); the SU(N-1)×U(1) covariance is already carried by the
# TensorMap, so vanishing of Φ (plus isometry) uniquely pins C up to the outer-multiplicity gauge.
function _coset_defect(C, gens)
    parts = Vector{Float64}[]
    for (Ja, Jb, Jc) in gens
        @tensor Φ[ad x y; z n] := Ja[ad x; x'] * C[x' y; z n] + Jb[ad y; y'] * C[x y'; z n] -
                                  C[x y; z' n] * Jc[ad z'; z]
        push!(parts, _cgc_coeffs(Φ))
    end
    return reduce(vcat, parts)
end

# Iterative refinement (roadmap Phase 3): the reduced descent accumulates ~1e-12 error at N≥4 from
# SU(N-1) F-symbol recoupling noise injected into every `@tensor` (the per-shell solves themselves are
# perfectly conditioned). The exact CGC is the unique isometry annihilated by the coset-intertwiner
# defect Φ, and the coset operators are exact (√ of rationals). So we project the descent output onto
# the true nullspace of Φ — assembled as a *compact* matrix over the isoscalar-factor coefficients (no
# densification) — then re-orthonormalize (polar). This restores ~1e-14 accuracy and, crucially, keeps
# the tower from compounding its own error. See research/phase3-gauge-handoff.md.
function _refine_reduced_CGC(CGC, a::I, b::I, c::I) where {I <: SUNIrrep}
    v0 = _cgc_coeffs(CGC)
    cdim = length(v0)
    cdim == 0 && return CGC
    gens = ((reduced_raising_operators(a), reduced_raising_operators(b), reduced_raising_operators(c)),
            (reduced_lowering_operators(a), reduced_lowering_operators(b), reduced_lowering_operators(c)))
    scratch = copy(CGC)
    ddim = length(_coset_defect(CGC, gens))
    M = zeros(Float64, ddim, cdim)
    e = zeros(Float64, cdim)
    for j in 1:cdim
        fill!(e, 0.0)
        e[j] = 1.0
        _cgc_setcoeffs!(scratch, e)
        M[:, j] = _coset_defect(scratch, gens)
    end
    F = svd(M)
    tol = TOL_GAUGE * (isempty(F.S) ? 1.0 : F.S[1])
    k = count(<(tol), F.S) + (cdim - length(F.S))       # nullity = Nsymbol(a,b,c)^2
    k == 0 && return CGC
    Ns = F.V[:, (cdim - k + 1):cdim]                    # orthonormal basis of the exact-CGC subspace
    refined = _cgc_setcoeffs!(copy(CGC), Ns * (Ns' * v0))
    U, _ = left_polar(refined)                          # polar factor: exact isometry, real
    return U
end

function reduced_CGC(a::I, b::I, c::I) where {I <: SUNIrrep}
    return get!(REDUCED_CGC_CACHE, (a, b, c)) do
        return _reduced_CGC(a, b, c)
    end
end

function _reduced_CGC(a::I, b::I, c::I) where {I <: SUNIrrep}
    Va = reduced_space(a)
    Vb = reduced_space(b)
    Vc = reduced_space(c)
    Vn = spacetype(Vc)(one(sectortype(Vc)) => Nsymbol(a, b, c))
    CGC = zeros(Va ⊗ Vb ← Vc ⊗ Vn)

    # highest weight: 0 = (Jpa ⊗ 1 + 1 ⊗ Jpb) * CG[hw]
    # by projecting onto the highest weight subspace (unique in fused basis?)
    P_hw = highest_weight_projector(Va ⊗ Vb, c)
    Jpa = reduced_raising_operators(a)
    Jpb = reduced_raising_operators(b)
    @tensor HW_eq[ad a b; c] := Jpa[ad a; a'] * P_hw[a' b; c] + Jpb[ad b; b'] * P_hw[a b'; c]

    # embed into entire space. `rtol` is needed because the coset-raising `HW_eq` accumulates
    # numerical error for larger irreps: the true null singular value can drift up to ~1e-11, well
    # above `rightnull`'s default cutoff, which would then miss the highest-weight coupling entirely
    # (empty `CG_w`, e.g. `15⊗15→1` at N=4). Genuine couplings are O(1), so a relative cut is safe.
    CG_w = P_hw * right_null(HW_eq; alg = :svd, trunc = (; rtol = 1.0e-9))'
    charge, bl = only(blocks(CG_w))
    # `gaugefix!` returns the orthonormalized basis as a new matrix; it does NOT write it back
    # into `bl` (it leaves `bl` in QR-mutated state). Must assign the return value, otherwise the
    # HW block is non-orthonormal whenever the fused HW sector is degenerate (blockdim > 1).
    block(CGC, charge) .= gaugefix!(bl)

    # lower weights: CG[w-1] * Jmc = (Jma ⊗ 1 + 1 ⊗ Jmb) * CG[w]
    Jma = reduced_lowering_operators(a)
    Jmb = reduced_lowering_operators(b)
    Jmc = reduced_lowering_operators(c)

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

    # Strip the descent's accumulated recoupling noise (~1e-12 at N≥4) by projecting onto the exact
    # coset-intertwiner nullspace + polar, before fixing the gauge on the now-accurate CGC.
    CGC = _refine_reduced_CGC(CGC, a, b, c)

    # Gauge-match the outer-multiplicity basis to the dense/GT `CGC` convention so the whole tower is
    # self-consistent: `reduced_CGC == dense CGC` (up to an F/R-irrelevant internal reorder), hence
    # F/R from a projector on `reduced_CGC` equal the dense F/R. The descent is linear in the outer-
    # multiplicity leg, so a single orthogonal `G` on that leg re-gauges every weight at once.
    G = _dense_gauge_transform(CGC, a, b, c)
    Gmap = TensorMap(G, Vn ← Vn)
    @tensor CGC_fixed[a b; c n] := CGC[a b; c n'] * Gmap[n'; n]
    return CGC_fixed
end
