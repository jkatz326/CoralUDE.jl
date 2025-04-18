using UniversalDiffEq
include("mumby_model.jl")
data, plt = coral_data(plot = true, σ1 = .1, σ2 = 0, u01 = .6, u02 = .1, λ = constant_fun(.3), datasize = 60, seed = 765, T = 50)
#model, test_data = ude_model_from_data(data)
#plt2 = phase_plane(model, step = .05)
#print(UniversalDiffEq.equilibrium_and_stability(model, .1, .9))
#plot(plt, plt2, layout = (1, 2), legend = false)
print(data)
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