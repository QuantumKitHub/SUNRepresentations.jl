struct GTPattern{N,L}
    data::NTuple{L,Int}
    function GTPattern{N}(data::NTuple{L,Int}) where {N,L}
        @assert 2 * L == N * (N + 1)
        return new{N,L}(data)
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

# GTPatternIterator: iterate over all GT-patterns associated to a given irrep
struct GTPatternIterator{N}
    irrep::SUNIrrep{N}
end

Base.IteratorSize(::Type{<:GTPatternIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:GTPatternIterator}) = Base.HasEltype()
Base.eltype(::GTPatternIterator{N}) where {N} = GTPattern{N,(N * (N + 1)) >> 1}
Base.length(iter::GTPatternIterator) = dim(iter.irrep)

function Base.iterate(iter::GTPatternIterator{1}, state=true)
    if state
        return GTPattern{1}((weight(iter.irrep)[1],)), false
    else
        return nothing
    end
end
function Base.iterate(iter::GTPatternIterator{N}) where {N}
    I = weight(iter.irrep)
    iter1 = Iterators.product(reverse(ntuple(i -> I[i + 1]:I[i], Val(N - 1)))...)
    next1 = Base.iterate(iter1)
    next1 === nothing && return nothing # should not happen
    v, state1 = next1
    iter2 = GTPatternIterator(SUNIrrep(reverse(v)))
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
        iter2 = GTPatternIterator(SUNIrrep(reverse(v)))
        next2 = Base.iterate(iter2)
        next2 === nothing && return nothing # should not happen
    end
    pat, state2 = next2
    newpat = GTPattern{N}((weight(iter.irrep)..., pat.data...))
    return newpat, (iter1, state1, iter2, state2)
end

highest_weight(irrep::SUNIrrep{1}) = GTPattern{1}(weight(irrep))
function highest_weight(irrep::SUNIrrep{N}) where {N}
    I = weight(irrep)
    d = highest_weight(SUNIrrep(Base.front(I)))
    return GTPattern{N}((I..., d.data...))
end

function creation(s::SUNIrrep{N}) where {N}
    d = dim(s)
    table = Dict{GTPattern,Int}(m => i for (i, m) in enumerate(basis(s)))
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

function cartan_operators(s::SUNIrrep{N}) where {N}
    d = dim(s)
    result = [SparseArray{RationalRoot{Int}}(undef, (d, d)) for i in 1:(N - 1)]
    for (i, m) in enumerate(basis(s))
        for (l, W) in enumerate(Zweight(m))
            result[l][i, i] = W
        end
    end
    return result
end
