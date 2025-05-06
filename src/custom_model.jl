using UniversalDiffEq
include("mumby_model.jl")
include("jld.jl")
covar_fun =  g -> g < .2 ? 0 : 1
λ = one_change_fun(.1, .3, 25)
model, test_data, inputs = ude_model(maxiter = 20, covar_fun = covar_fun, return_covar = true, return_inputs = true)
save_model("test_file", model, inputs)
model1, test_data1 = load_model("test_file")
state_estimates(model1)
#print(UniversalDiffEq.equilibrium_and_stability(model, .1, .9))
#plot(plt, plt2, layout = (1, 2), legend = false)
#=
DataFrames = ">=1.7.0"
DiffEqFlux = "4.1.0"
Distributions = ">=0.25.116"
FFMPEG = ">=0.4.2"
JLD2 = ">=0.5.11"
LLVM = ">=9.0.0"
Lux = "0.5 - 1.4"
OrdinaryDiffEq = ">=6.91.0"
Plots = ">=1.40.9"
Random = ">=1.11.0"
StochasticDiffEq = ">=6.69.1"
julia = ">=1.11"
=#