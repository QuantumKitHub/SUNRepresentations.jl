#=
Functionality for writing SU(N) CGCs in terms of SU(N-1) x U(1)
=#

# Compute SU(N-1) by discarding the first row and renormalizing (make last entry 0)
# Compute U(1) by how far down the second row is from the first, but shift such that
# total weight in the multiplet is 0
function reduced_charge(g::GTPattern{N}) where {N}
    I = _normalize(SUNIrrep(ntuple(i -> g[i, N - 1], N - 1)))
    Y = U1Irrep(N * rowsum(g, N - 1) - (N - 1) * rowsum(g, N))
    if N == 3
        I = SU2Irrep(I.I[1] // 2)
    end
    return I ⊠ Y
end

function reduced_space(a::SUNIrrep{N}) where {N}
    I = (N == 3 ? SU2Irrep : SUNIrrep{N - 1}) ⊠ U1Irrep
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

adjoint_irrep(::SUNIrrep{N}) where {N} = _normalize(SUNIrrep{N}(2, 1, ntuple(Returns(0), N - 2)...))
adjoint_irrep(::SUNIrrep{2}) = SUNIrrep(2, 0)
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
    g = TensorMap(g_array, Vad ⊗ Va ← Va; tol)
    return g
end

function reduced_raising_operators(a::SUNIrrep{3}; tol = 1.0e-12)
    p = reduced_basistransform(a)
    Jp1, Jp2 = map(x -> x[p, p], creation(a))
    Jp3 = commutator(Jp1, Jp2)
    raising_generators = stack((-Jp3, Jp2); dims = 1)
    
    Va = reduced_space(a)
    Vad_raising = spacetype(Va)((1 // 2, -3) => 1)

    g = TensorMap(raising_generators, Vad_raising ⊗ Va ← Va; tol)
    return g
end

function reduced_lowering_operators(a::SUNIrrep{3}; tol = 1.0e-12)
    p = reduced_basistransform(a)
    Jm1, Jm2 = map(x -> x[p, p], annihilation(a))
    Jm3 = commutator(Jm1, Jm2)
    lowering_generators = stack((-Jm2, Jm3); dims = 1)

    Va = reduced_space(a)
    Vad_lowering = spacetype(Va)((1 // 2, 3) => 1)
    g = zeros(Vad_lowering ⊗ Va ← Va)

    g = TensorMap(lowering_generators, Vad_lowering ⊗ Va ← Va; tol)
    return g
end

function weight_space(V, a::SUNIrrep, w::Integer)
    I = (a.N == 3 ? SU2Irrep : SUNIrrep{a.N - 1}) ⊠ U1Irrep
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


function reduced_CGC(a::I, b::I, c::I) where {I <: SUNIrrep}
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

    # embed into entire space
    CG_w = P_hw * rightnull(HW_eq)'
    charge, bl = only(blocks(CG_w))
    gaugefix!(bl)
    block(CGC, charge) .= bl 
    
    # lower weights: CG[w-1] * Jmc = (Jma ⊗ 1 + 1 ⊗ Jmb) * CG[w]
    Jma = reduced_lowering_operators(a)
    Jmb = reduced_lowering_operators(b)
    Jmc = reduced_lowering_operators(c)
  
    Pw = highest_weight_projector(Vc, c)
    while true
        @tensor begin
            CGCw[a b; c n] := CGC[a b; c' n] * Pw[c'; c]
            rhs[a b n; ad c] := Jma[ad a; a'] * CGCw[a' b; c n] + Jmb[ad b; b'] * CGCw[a b'; c n]
            eqs[c; ad c'] := Jmc[ad c; c''] * Pw[c''; c']
        end
        @tensor CGC[a b; c n] += (rhs / eqs)[a b n; c]
        Pw = isometry(Vc, infimum(Vc, fuse(domain(Pw) ⊗ space(Jmc, 1)')))
        dim(Pw) == 0 && break
    end
    return CGC
end
