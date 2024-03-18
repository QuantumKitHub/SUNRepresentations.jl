using Pkg: Pkg;
Pkg.instantiate();

using ThreadPinning
ThreadPinning.pinthreads(:cores)
ThreadPinning.threadinfo(;blas=true, hints=true)

using BenchmarkTools
using Random
using Test
using LinearAlgebra: qr!, ldiv!
using TensorKit
using SUNRepresentations
using SUNRepresentations: trivial_CGC, highest_weight_CGC, lower_weight_CGC!, weightmap,
                          _emptyindexlist
using CairoMakie
using DataFrames, Statistics

# Function to benchmark
# ---------------------

function rand_sectors(N::Int, maxnum::Int=50)
    I = SUNIrrep{N}
    s1 = rand(collect(Iterators.take(values(I), maxnum)))
    s2 = rand(collect(Iterators.take(values(I), maxnum)))
    s3 = rand(collect(s1 ⊗ s2))
    return s1, s2, s3
end

_CGC(T::Type{<:Real}, s1, s2, s3, ::Val{0}) = SUNRepresentations._CGC(T, s1, s2, s3)
function _CGC(T::Type{<:Real}, s1, s2, s3, mode::Val{N}) where {N}
    if isone(s1)
        @assert s2 == s3
        CGC = trivial_CGC(T, s2, true)
    elseif isone(s2)
        @assert s1 == s3
        CGC = trivial_CGC(T, s1, false)
    else
        CGC = highest_weight_CGC(T, s1, s2, s3)
        _lower_weight_CGC!(mode, CGC, s1, s2, s3)
    end
    @debug "Computed CGC: $s1 ⊗ $s2 → $s3"
    return CGC
end

include("implementations.jl")

# Generate benchmarks
# -------------------

const suite = BenchmarkGroup()

CGC_benchmarks = suite["CGC"]

const T = Float64
const MAXNUM = 50
const NUM_TESTS = 100
const MAX_SUN = 5
const MODE = 2

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 60

for N in 3:MAX_SUN, _ in 1:NUM_TESTS
    s1, s2, s3 = rand_sectors(N, MAXNUM)
    @info "$s1 ⊗ $s2 → $s3"

    base_cgc = _CGC(T, s1, s2, s3, Val(0))
    CGC_benchmarks[0][s1, s2, s3] = @benchmarkable _CGC($T, $s1, $s2, $s3, Val(0))

    new_cgc = _CGC(T, s1, s2, s3, Val(2))
    @test new_cgc ≈ base_cgc
    CGC_benchmarks[MODE][s1, s2, s3] = @benchmarkable _CGC($T, $s1, $s2, $s3, Val(MODE))
end

results = run(CGC_benchmarks; verbose=true)
BenchmarkTools.save("benchmark_results.json", results)

# Process results
# ---------------

estimator = minimum
begin
    df = DataFrame(; s1=SUNIrrep[], s2=SUNIrrep[], s3=SUNIrrep[], mode=Int[],
                   t_min=Float64[], a_min=Int[], m_min=Float64[])

    for (mode, bench) in results
        for ((s1_str, s2_str, s3_str), trial) in bench
            s1 = eval(Meta.parse(s1_str))
            s2 = eval(Meta.parse(s2_str))
            s3 = eval(Meta.parse(s3_str))
            t_min = estimator(trial).time
            a_min = estimator(trial).allocs
            m_min = estimator(trial).memory

            push!(df, (; s1, s2, s3, mode, t_min, a_min, m_min))
        end
    end
end

# Report results
# --------------

timing_tick_vals = [-6.0, -3.0, 0.0, log10(60.0), log10(3600.0)]
timing_tick_labels = ["1μs", "1ms", "1s", "1m", "1h"]
colors = Makie.wong_colors()

function plot_timing!(ax, df)
    df_filtered = select(df, :s1, :s2, :s3, :t_min, :mode)
    df_joined = innerjoin(groupby(df_filtered, :mode; sort=true)...; on=[:s1, :s2, :s3],
                          renamecols="_old" => "_new")
    ax.title = "Timing comparison"
    ax.xlabel = "baseline timing"
    ax.xticks = (timing_tick_vals, timing_tick_labels)
    ax.ylabel = "Speedup factor"

    scatter!(ax, log10.(df_joined.t_min_old) .- 9,
             df_joined.t_min_old ./ df_joined.t_min_new;
             color=colors[getproperty.(df_joined.s1, :N)])
    return ax
end

function plot_allocations!(ax, df)
    df_filtered = select(df, :s1, :s2, :s3, :a_min, :mode)
    df_joined = innerjoin(groupby(df_filtered, :mode; sort=true)...; on=[:s1, :s2, :s3],
                          renamecols="_old" => "_new")
    ax.title = "Allocation comparison"
    ax.xlabel = "baseline allocations"
    ax.ylabel = "Speedup factor"
    ax.xtickformat = values -> ["1e$value" for value in values]

    scatter!(ax, log10.(df_joined.a_min_old), df_joined.a_min_new ./ df_joined.a_min_old;
             color=colors[getproperty.(df_joined.s1, :N)])
    return ax
end

function plot_memory!(ax, df)
    df_filtered = select(df, :s1, :s2, :s3, :m_min, :mode)
    df_joined = innerjoin(groupby(df_filtered, :mode; sort=true)...; on=[:s1, :s2, :s3],
                          renamecols="_old" => "_new")
    ax.title = "Memory comparison"
    ax.xlabel = "baseline memory"
    ax.ylabel = "Speedup factor"
    ax.xticks = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 ["1B", "10B", "100B", "1KB", "10KB", "100KB", "1MB", "10MB", "100MB",
                  "1GB"])
    scatter!(ax, log10.(df_joined.m_min_old), df_joined.m_min_new ./ df_joined.m_min_old;
             color=colors[getproperty.(df_joined.s1, :N)])
    return ax
end

function plot_comparison!(f, df)
    gf = f[1, 1] = GridLayout(1, 3)
    plot_timing!(Axis(gf[1, 1]), df)
    plot_allocations!(Axis(gf[1, 2]), df)
    plot_memory!(Axis(gf[1, 3]), df)
    return f
end

f = let f = Figure(; size=(1200, 800))
    plot_comparison!(f, df)
    Legend(f[1, 1][1, 2], [PolyElement(; color=colors[N]) for N in 3:MAX_SUN],
           [string("SU($N)") for N in 3:MAX_SUN]; tellwidth=false, tellheight=false,
           margin=(10, 10, 10, 10), halign=:center, valign=:top)
    f
end

save("benchmark_results.png", f; px_per_unit=2)