using CoralUDE
using Test

@testset "CoralUDE.jl" begin
    @test CoralUDE.greet_CoralUDE() == "Hello CoralUDE!"
    @test CoralUDE.greet_CoralUDE() != "Hello world!"
    @test try
        CoralUDE.testUDE()
        true
    catch
        false
    end
    @test try
        CoralUDE.coral_data()
        println("coral_data passed")
        true
    catch
        println("coral_data failed")
        false
    end
    @test try
        CoralUDE.ude_model()
        println("ude_model passed")
        true
    catch
        println("ude_model failed")
        false
    end
    @test try
        data = CoralUDE.coral_data()
        CoralUDE.ude_model_from_data(data)
        println("ude_model_from_data passed")
        true
    catch
        println("ude_model_from_data failed")
        false
    end
    @test try
        CoralUDE.node_model()
        println("node_model passed")
        true
    catch
        println("node_model failed")
        false
    end
    @test try
        model, test_data = CoralUDE.ude_model()
        CoralUDE.state_estimates(model)
        println("state_estimates passed")
        true
    catch
        println("state_estimates failed")
        false
    end
    @test try
        model, test_data = CoralUDE.ude_model()
        CoralUDE.phase_plane(model)
        println("phase_plane passed")
        true
    catch
        println("phase_plane failed")
        false
    end
end
