module CoralUDE

# Write your package code here.
include("functions.jl")
include("mumby_model.jl")
include("jld.jl")
export greet_CoralUDE, coral_data, ude_model_from_data, ude_model, node_model, phase_plane, state_estimates, save_rhs, load_rhs, make_sparse, save_model, load_model, combine_data_ude_inputs

end
