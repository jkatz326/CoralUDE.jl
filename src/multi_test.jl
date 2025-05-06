using UniversalDiffEq
include("jld.jl")
include("mumby_model_multi.jl")
#=
data1, plt1 = coral_data(plot = true, σ1 = 0, σ2 = 0, u01 = .3, u02 = .2, λ = constant_fun(.3), datasize = 60, seed = 765, T = 150)
data2, plt2 = coral_data(plot = true, σ1 = 0, σ2 = 0, u01 = .6, u02 = .1, λ = constant_fun(.3), datasize = 60, seed = 765, T = 150)
combined = merge_coral_data(data1, data2)
model, test_data = ude_model_from_data(combined)
save_rhs("test_rhs2", model)
rhs = get_right_hand_side(model)
print(rhs)
=#
rhs = load_rhs("test_rhs2")
print(rhs([.3, .2], 1, 5))
