struct GTPattern{N, L}
    data::NTuple{L, Int}
    function GTPattern{N}(data::NTuple{L, Int}) where {N, L}
        @assert 2 * L == N * (N + 1)
        return new{N, L}(data)
    end
end

Base.getproperty(m::GTPattern{N}, f::Symbol) where {N} = f == :N ? N : getfield(m, f)

Base.checkbounds(m::GTPattern, k, l) = 0 < k <= l <= m.N

@inline function Base.getindex(m::GTPattern{N}, k, l) where {N}
    @boundscheck begin
        checkbounds(m, k, l) || throw(BoundsError(m, (k, l)))
    end
    v = @inbounds m.data[k + (((l + 1 + N) * (N - l)) >> 1)]
    return v
end
@inline function Base.setindex(m::GTPattern{N}, v, k, l) where {N}
    @boundscheck begin
        (0 < k <= l <= N) || throw(BoundsError(m, (k, l)))
    end
    newdata = Base.setindex(m.data, v, k + (((l + 1 + N) * (N - l)) >> 1))
    return GTPattern{N}(newdata)
end

function Base.show(io::IO, m::GTPattern)
    N = m.N
    if get(io, :compact, false)
        return println(io, "GTPattern{$N}(", m.data, ")")
    end
    println(io, "GTPattern{", N, "}:")
    width = ndigits(maximum(m.data))
    s = repeat(" ", width)
    for i in 1:N
        if i == 1
            print(io, N == 1 ? "( " : "⎛ ")
        elseif i == N
            print(io, "⎝ ")
        else
            print(io, "⎜ ")
        end
        print(io, repeat(s, i - 1))
        l = N - i + 1
        for k in 1:(l - 1)
            print(io, repeat(" ", width - ndigits(m[k, l])), m[k, l], s)
        end
        print(io, repeat(" ", width - ndigits(m[l, l])), m[l, l])
        print(io, repeat(s, i - 1))
        if i == 1
            println(io, N == 1 ? " )" : " ⎞")
        elseif i == N
            println(io, " ⎠")
        else
            println(io, " ⎟")
        end
    end
    return
end
function Base.isless(ma::GTPattern{N}, mb::GTPattern{N}) where {N}
    @inbounds for l in N:-1:1, k in 1:l
        mb[k, l] > ma[k, l] && return true
        mb[k, l] < ma[k, l] && return false
    end
    return false
end
Base.hash(m::GTPattern, h::UInt) = hash(m.data, hash(m.N, h))
Base.:(==)(ma::GTPattern, mb::GTPattern) = ma.data == mb.data
rowsum(m::GTPattern, l) = l == 0 ? 0 : sum(m[k, l] for k in 1:l)

#the Z weight
Zweight(m::GTPattern) = Z2weight(m) .// 2
function Z2weight(m::GTPattern{N}) where {N}
    let w = (0, ntuple(l -> rowsum(m, l), Val(N))...)
        return ntuple(Val(N - 1)) do l
            return 2 * w[l + 1] - (w[l] + w[l + 2])
        end
    end
end

#the pattern weight
function weight(m::GTPattern{N}) where {N}
    w = ntuple(l -> rowsum(m, l), Val(N))
    @inbounds for l in N:-1:2
        w = Base.setindex(w, w[l] - w[l - 1], l)
    end
    return w
end

# GTPatternIterator: iterate over all GT-patterns associated to a given irrep.
# Stores the top-row weight values directly as an NTuple{N, Int} to avoid going
# through SUNIrrep normalisation for intermediate (possibly un-normalised) sub-rows.
struct GTPatternIterator{N}
    toprow::NTuple{N, Int}
end

basis(s::SUNIrrep{N}) where {N} = GTPatternIterator{N}(weight(s))

Base.IteratorSize(::Type{<:GTPatternIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:GTPatternIterator}) = Base.HasEltype()
Base.eltype(::GTPatternIterator{N}) where {N} = GTPattern{N, (N * (N + 1)) >> 1}
# dim is shift-invariant (uses only differences), so the weight constructor gives the
# same result regardless of whether toprow is normalised.
Base.length(iter::GTPatternIterator{N}) where {N} = dim(SUNIrrep{N}(iter.toprow))

function Base.iterate(iter::GTPatternIterator{1}, state = true)
    state || return nothing
    return GTPattern{1}((iter.toprow[1],)), false
end
function Base.iterate(iter::GTPatternIterator{N}) where {N}
    I = iter.toprow
    iter1 = Iterators.product(reverse(ntuple(i -> I[i + 1]:I[i], Val(N - 1)))...)
    next1 = Base.iterate(iter1)
    next1 === nothing && return nothing # should not happen
    v, state1 = next1
    iter2 = GTPatternIterator{N - 1}(reverse(v))
    next2 = Base.iterate(iter2)
    next2 === nothing && return nothing # should not happen
    pat, state2 = next2
    newpat = GTPattern{N}((I..., pat.data...))
    return newpat, (iter1, state1, iter2, state2)
end
function Base.iterate(iter::GTPatternIterator{N}, state) where {N}
    (iter1, state1, iter2, state2) = state
    next2 = Base.iterate(iter2, state2)
    if next2 === nothing
        next1 = Base.iterate(iter1, state1)
        next1 === nothing && return nothing
        v, state1 = next1
        iter2 = GTPatternIterator{N - 1}(reverse(v))
        next2 = Base.iterate(iter2)
        next2 === nothing && return nothing # should not happen
    end
    pat, state2 = next2
    newpat = GTPattern{N}((iter.toprow..., pat.data...))
    return newpat, (iter1, state1, iter2, state2)
end

# Build the highest-weight GT pattern from a top row of actual weight values.
_highest_weight_from_row(row::NTuple{1, Int}) = GTPattern{1}(row)
function _highest_weight_from_row(row::NTuple{N, Int}) where {N}
    sub = _highest_weight_from_row(Base.front(row))
    return GTPattern{N}((row..., sub.data...))
end

highest_weight(irrep::SUNIrrep) = _highest_weight_from_row(weight(irrep))

function creation(s::SUNIrrep{N}) where {N}
    d = dim(s)
    table = Dict(m => i for (i, m) in enumerate(basis(s)))
    result = [SparseArray{RationalRoot{Int}}(undef, (d, d)) for i in 1:(N - 1)]
    @inbounds for (m, i) in table
        for l in 1:(N - 1), k in 1:l
            coef = -1 // 1
            for k′ in 1:(l + 1)
                coef *= m[k′, l + 1] - m[k, l] + k - k′
                if k′ <= l - 1
                    coef *= m[k′, l - 1] - m[k, l] + k - k′ - 1
                end
                numerator(coef) == 0 && break
                if k′ <= l && k′ != k
                    coef //= (m[k′, l] - m[k, l] + k - k′) *
                        (m[k′, l] - m[k, l] + k - k′ - 1)
                end
                denominator(coef) == 0 && break
            end
            (denominator(coef) == 0 || numerator(coef) == 0) && continue
            m′ = Base.setindex(m, m[k, l] + 1, k, l)
            j = table[m′]
            result[l][j, i] = signedroot(coef)
        end
    end
    return result
end

annihilation(s::SUNIrrep) = [SparseArray(op') for op in creation(s)]

# Littlewood-Richardson rule in GT form: enumerate the GT patterns m of λ whose tableau, added to μ
# in reverse reading order, keeps t = μ + content a partition. Tableau row k holds m[k, N:-1:k],
# with m[k, l] - m[k, l - 1] boxes of entry l; all constraints are imposed as bounds on m[k, l - 1].
# Returns the number of patterns ending in `target` (any if `nothing`), and records the final `t`
# of each in `out`. `cur` and `prev` hold rows k and k - 1 of m.
function littlewoodrichardson!(out, s1::I, s2::I, target = nothing) where {N, I <: SUNIrrep{N}}
    λ, μ = weight(s1), weight(s2)
    cur = Base.setindex(ntuple(Returns(0), Val(N)), λ[1], N)
    return littlewoodrichardson!(out, λ, target, μ, cur, cur, 1, N)
end
function littlewoodrichardson!(
        out, λ::NTuple{N, Int}, target, t::NTuple{N, Int},
        prev::NTuple{N, Int}, cur::NTuple{N, Int}, k::Int, l::Int
    ) where {N}
    if l == k # boxes of entry k end row k
        t = Base.setindex(t, t[k] + cur[k], k)
        if k == N - 1 # row N is empty, since λ[N] == 0
            record!(out, t)
            return 1
        end
        return littlewoodrichardson!(out, λ, target, t, cur, Base.setindex(cur, λ[k + 1], N), k + 1, N)
    end

    # choose v = m[k, l - 1], which adds cur[l] - v boxes of entry l
    lo = max(λ[k + N - l + 1], t[l] + cur[l] - t[l - 1]) # GT bound, lattice condition
    hi = k > 1 ? min(cur[l], prev[l - 2]) : cur[l] # GT interleaving
    isnothing(target) || (lo = max(lo, cur[l] - (target[l] - t[l])))
    if l - 1 == k # v = m[k, k] is the number of boxes of entry k
        k > 1 && (hi = min(hi, t[k - 1] - t[k]))
        if !isnothing(target) # no boxes of entry k after row k
            lo = max(lo, target[k] - t[k])
            hi = min(hi, target[k] - t[k])
        end
    end

    n = 0
    for v in lo:hi
        t′ = Base.setindex(t, t[l] + cur[l] - v, l)
        n += littlewoodrichardson!(out, λ, target, t′, prev, Base.setindex(cur, v, l - 1), k, l - 1)
    end
    return n
end

# record the weight `t` of a term of the direct product: not at all, as an irrep, or as a count
record!(::Nothing, t) = nothing
record!(out::AbstractVector{I}, t) where {I} = (push!(out, I(t)); nothing)
function record!(out::AbstractDict{I, Int}, t) where {I}
    s = I(t)
    out[s] = get(out, s, 0) + 1
    return nothing
end

# direct product: return dictionary with new irreps as keys, outer multiplicities as value
function directproduct(s1::I, s2::I) where {I <: SUNIrrep}
    nboxes(s1) > nboxes(s2) && return directproduct(s2, s1)
    counts = Dict{I, Int}()
    littlewoodrichardson!(counts, s1, s2)
    return counts
end

# direct product as a sorted vector of irreps and outer multiplicities
function sorteddirectproduct(s1::I, s2::I) where {N, I <: SUNIrrep{N}}
    nboxes(s1) > nboxes(s2) && return sorteddirectproduct(s2, s1)
    irreps = sizehint!(I[], min(dim(s1), 512)) # the number of terms is at most dim(s1)
    littlewoodrichardson!(irreps, s1, s2)

    # counting in a dictionary only pays off for large products with many multiplicities
    if length(irreps) > 512
        counts = Dict{I, Int}()
        for s in irreps
            counts[s] = get(counts, s, 0) + 1
        end
        return sort!(collect(counts); by = first)
    end

    sort!(irreps)
    result = Pair{I, Int}[]
    for s in irreps
        if !isempty(result) && first(last(result)) == s
            result[end] = s => last(result[end]) + 1
        else
            push!(result, s => 1)
        end
    end
    return result
end
