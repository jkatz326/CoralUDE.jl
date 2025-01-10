using UniversalDiffEq, DataFrames
function greet_CoralUDE()
    return "Hello CoralUDE!"
end

function testUDE()
    data,plt = LotkaVolterra();
    model = NODE(data);
    gradient_descent!(model);
    plot_predictions(model)
    plot_state_estimates(model)
end 

export greet_CoralUDE