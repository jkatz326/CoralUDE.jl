using CoralUDE
using Test

@testset "CoralUDE.jl" begin
    @testset "greet_CoralUDE" begin 
        @test CoralUDE.greet_CoralUDE() == "Hello CoralUDE!"
        @test CoralUDE.greet_CoralUDE() != "Hello world!"
    end 

    @testset "testUDE" begin 
        @test try
            CoralUDE.testUDE()
            true
        catch
            false
        end
    end 

    @testset "coral_data" begin
        @test try
            CoralUDE.coral_data()
            true
        catch
            false
        end
    end 

    @testset "ude_model" begin
        @test try
            CoralUDE.ude_model(maxiter = 100)
            true
        catch
            false
        end
    end 

    @testset "ude_model_from_data" begin 
        @test try
            data = CoralUDE.coral_data()
            CoralUDE.ude_model_from_data(data, maxiter = 100)
            true
        catch
            false
        end
    end 

    @testset "node_model" begin
        @test try
            CoralUDE.node_model(maxiter = 100)
            true
        catch
            false
        end
    end 

    @testset "state_estimates" begin 
        @test try
            model, test_data = CoralUDE.ude_model(maxiter = 50)
            CoralUDE.state_estimates(model)
            true
        catch
            false
        end
    end 

    @testset "phase_plane" begin 
        @test try
            model, test_data = CoralUDE.ude_model(maxiter = 50)
            CoralUDE.phase_plane(model)
            true
        catch
            false
        end
    end 
    
end
