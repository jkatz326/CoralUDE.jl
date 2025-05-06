using UniversalDiffEq, DataFrames, Plots

# oscliating data from 2 sites at 20 time points 
dat = DataFrame(t = vcat(collect(1:20),collect(1:20)), 
                series = vcat(ones(20),2*ones(20)),
                x1 = sin.(1:40), x2 = sin.((2.5:1:41.5)))

Plots.plot(dat.t,dat.x1, group = dat.series , c = 1)
Plots.plot!(dat.t,dat.x2, group = dat.series, c=2)

# neural network with two inputs and two outputs
NN,NNparams = SimpleNeuralNetwork(2,2)
init_parameters = (NN=NNparams, )
function derivs!(u,i,p,t)
    return NN(u,p.NN)
end 

ude = MultiCustomDerivatives(dat,derivs!,init_parameters)

train!(ude, loss_function = "derivative matching", optim_options = (maxiter = 500, step_size = 0.01), 
        verbose = true, regularization_weight = 0.0)

# plot in sample predictions
plot_state_estimates(ude)
plot_predictions(ude)

# compare with out of sample data
dat_test = DataFrame(t = collect(31:40), series = ones(10),
                x1 = sin.(31:40), x2 = sin.((32.5:1.0:41.5)))
plot_forecast(ude,dat_test )

# get ODE functions 
function get_right_hand_side_(ude)
    (u,i,t) -> ude.process_model.rhs(u,i,ude.parameters.process_model,t)
end 
rhs = get_right_hand_side_(ude)


# simulation of trained ode model  
dt=0.05
ut = [1.2,0.9]
T = 10
times = 0:dt:T
u = zeros(2,length(times))
i = 0
for t in times
    i +=1
    ut .+= dt * rhs(ut,1,0.0)
    u[:,i] .= ut
end
Plots.plot(times,u', label = "", width = 2, xlabel = "time")