module SUNClebschGordan
    #TensorKit dependency will be dropped in future
	using TensorKit,TensorOperations,RowEchelon
    import LinearAlgebra


    export SUNIrrep,CGC;

    TO = TensorOperations;

    include("auxiliary.jl");
    include("sector.jl");
    include("gtp.jl");
    include("cgc.jl");

end # module
