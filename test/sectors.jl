for I in sectorlist
    println("------------------------------------")
    println("Sector $I")
    println("------------------------------------")
    @testset "Sector $I: Additional basic properties" begin
        s = (randsector(I), randsector(I), randsector(I))

        mode_old = SUNRepresentations.display_mode("dimension")
        for mode in ["dimension", "dynkin", "weight"]
            SUNRepresentations.display_mode(mode)
        end
        SUNRepresentations.display_mode(mode_old)
        for i in 1:3
            @test 1 == @constinferred Nsymbol(s[i], dual(s[i]), unit(s[i]))
        end
    end

    try
        s = sprint(SUNRepresentations.cache_info)
    catch
        @test false
    end
end
