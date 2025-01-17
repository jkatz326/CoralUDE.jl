using UniversalDiffEq, Lux, DataFrames, OrdinaryDiffEq, Distributions, Random, Plots

export coral_data, ude_model_from_data, ude_model, node_model, phase_plane, state_estimates

#HELPER FUNCTIONS 

#=Grazing Rate Functions: In our model, the grazing parameter g is replaced by a function λ(t) that takes in
a value of time and outputs. The four functions below take in various values and output a function λ(t).
=#
 
#Exponential growth or decay from v1 to v2 governed by rate
function exp_fun(v1, v2, rate)
    if v1 > v2
        return t -> (v1 - v2) * exp(-rate * t) + v2 #exp decay 
    else  
        return t -> v2 / (1 + ((v2 - v1)/v1) * exp(-rate * t)) #logistic growth
    end
end
	
#Function that outputs v1 always
function constant_fun(v1)
    return t -> v1
end 
	
#Function that outputs v1 before time tswitch and v2 afterwards
function one_change_fun(v1, v2, tswitch)
    return t -> t < tswitch ? v1 : v2 
end

#Function that switches from v1 to v2 at time t1 but switches back to v1 at time t2
function pulse_function(v1, v2, t1, t2)
	if t1 < t2 
		return t -> (t < t1) || (t >= t2) ? v1 : v2
	else 
		return t -> t < t1 ? v1 : v2 #If t2 is before or same time as t1, return one change function
	end 
end 

#Given a set of parameters p and a time t, computes the grazing rate at time t and adds process noise to each parameter
function unpack_values(p, t)
	dist = p.dist
	noise = rand(dist, 1)
	return p.a, p.γ, p.r, p.d, p.λ(t) + noise[1]
end;

#Ensures that initial conditions are not too close to 0 or 1
function valid_range(u)
	return max(min(1 - 1e-4, u[1]), 1e-4), max(min(1 - 1e-4, u[2]), 1e-4)
end;

#= 
Interpretation of ODEs below (Mumby 2007):
- https://www.nature.com/articles/nature06252
- M (u1), C(u2), and T represent the percentage of Seabed that is dominated by Macroalgae, Coral, and Algal Turf respectively.
- Since M + C + T = 1, we have that T = 1 - M - C.
- Thus, we can describe the entire system with two ODEs: dM/dt and dC/dt.
- These ODEs are governed by the following five parameters:
- 'a' controls the rate at which macroalgae grow over coral. 
- 'g' controls the rate at which parrotfish and other animals graze macroalgae. Under the hood, g is a function of time λ.
- 'γ' controls the rate at which macroalgae grow over algal turfs.
- 'r' controls the rate at which coral grows over algal turfs.
- 'd' controls the rate of natural coral mortality.
- The rate "- (g * M)/(M + T)" represents the assumption that fish are equally likely to graze Macroalgae and Algal Turf.
- So, the change of Macroalgae depends on the grazing factor times the percentage of Total Algae that is Macroalgae.
- All other dependencies in the ODEs below are more intuitive. 
=#
function mumby_eqns(du, u, p, t)
    u1, u2 = valid_range(u)
	a, γ, r, d, g = unpack_values(p, t)
    du[1] = a*u1*u2 - (g*u1)/(1 - u2) + γ*u1*(1 - u1 - u2)
    du[2] = r*(1 - u1 - u2)*u2 - d*u2 - a*u1*u2
end;

#Helper function for determining how effective forecasts are
function when_below(data, val)
	for row in eachrow(data)
		if (row.x2 < val)
			return row.time
		end 
	end 
	return - 1
end

# EXPORTED FUNCTIONS 

#Synthetic Data Generation:
#=
Using the ODEs described above, generates a model of the equations where the death rate term is replaced by a neural network. 
This neural network is trained on a subset of the synthetic data that is designated for training. 
Then, the function outputs the trained model along with the remaining data for testing.
Parameters-- 
- plot : true generates a plot of the data
- seed : random seed for data Generation
- σ1 : parameter that controls process noise
- σ2 : parameter that controls observation noise
- The remaining paramters are from the Mumby model. See above. 
=#
function coral_data(;plot = false, seed = 123, datasize = 60, T = 300, σ1 = 0, σ2 = 0, u01 = .2, u02 = .2, a = .1, γ = .8, r = 1, d = .44, λ = constant_fun(.3))
    # set seed 
    Random.seed!(seed)

    # set parameters
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[u01, u02]
    p = (a = a, γ = γ, r = r, d = d, λ = λ, dist = Normal(0.0, σ1))

    # generate time series with DifferentialEquations.jl
    prob_trueode = ODEProblem(mumby_eqns, u0, tspan, p)
    ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

    # add observation noise 
    ode_data .+= ode_data .* rand(Normal(0.0, σ2), size(ode_data))

    if plot
        plt = Plots.scatter(tsteps,transpose(ode_data), xlabel = "Time", ylabel = ["Macroalgae % Cover", "Coral % Cover"], label = "")
        data = data = DataFrame(x1 = ode_data[1,:], x2 = ode_data[2,:], time = tsteps)
        return data, plt
    end
    data = DataFrame(x1 = ode_data[1,:], x2 = ode_data[2,:], time = tsteps)
    return data
end

#Universal Differential Equation Model:
#=
Using the ODEs described above, generates a model of the equations where the death rate term is replaced by a neural network. 
This neural network is trained on a subset of the synthetic data that is designated for training. 
Then, the function outputs the trained model along with the remaining data for testing.
Parameters-- 
- speedy = false allows for complete training of model. When speedy = true, training cut off after 50 iterations.
- Training data will be generated from times 0 to T1
- Testing data will be generated from times T1 + 1 to T2
- Description of other model parameters can be found in code block above. 
=#

function ude_model_from_data(data;speedy = false, T1 = 175, T2 = 300, u01 = .2, u02 = .2, a = .1, γ = .8, r = 1, d = .44)

	#Seperate both data frames into testing data and training data 
    train_data = subset(data, :time => time -> time .<= T1)
    test_data = subset(data, :time => time -> time .> T1)

    # set neural network dimensions
    dims_out = 1
    dims_in = 3
    hidden = 5

    # Define neural network with Lux.jl
    NN = Lux.Chain(Lux.Dense(dims_in, hidden, tanh), Lux.Dense(hidden, dims_out))
    rng = Random.default_rng()
    NNparameters, NNstates = Lux.setup(rng,NN)
    parameters = (NN = NNparameters, a = a, γ = γ, r = r, d = d)
	
    # Define derivatives (time dependent NODE)
    function derivs!(du,u,p,t)
        vals = NN([u[1],u[2],t],p.NN,NNstates)[1]
		u1, u2 = valid_range(u)
		a, γ, r, d = p.a, p.γ, p.r, p.d
        du[1] = a*u1*u2 - vals[1] + γ*u1*(1 - u1 - u2)
        du[2] = r*(1 - u1 - u2)*u2 - d*u2 - a*u1*u2
        return du
    end

	#Generate model using UniversalDiffEq
    model = CustomDerivatives(train_data, derivs!,parameters; proc_weight = 2.5, obs_weight = 1, reg_weight = 10^-5, reg_type = "L2")

	#Train the model. If speedy, cut off training early. 
    if (speedy)
        gradient_descent!(model, maxiter = 50)
    else 
        gradient_descent!(model)
    end
    return model, test_data
end

#Wrapper around ude_model_from_data that generates the data before training the model.
function ude_model(;speedy = false, σ1 = 0, σ2 = 0, T1 = 175, T2 = 300, u01 = .2, u02 = .2, a = .1, γ = .8, r = 1, d = .44, λ = constant_fun(.3))
	#Generete Synthetic data using Mumby Equations
    data = coral_data(T = T2, σ1 = σ1, σ2 = σ2, u01 = u01, u02 = u02, a = a, γ = γ, r = r, d = d, λ = λ)
	
    return ude_model_from_data(data, speedy = speedy, T1 = T1, T2 = T2, u01 = u01, u02 = u02, a = a, γ = γ, r = r, d = d)
end

#Same as the UDE function above but trains a generic neural net [NODE] rather than a UDE
function node_model(;speedy = false, T1 = 175, T2 = 300, σ1 = 0, σ2 = 0, u01 = .2, u02 = .2, a = .1, γ = .8, r = 1, d = .44, λ = constant_fun(.3))
    #Generete Synthetic data using Mumby Equations
    data = coral_data(T = T2, σ1 = σ1, σ2 = σ2, u01 = u01, u02 = u02, a = a, γ = γ, r = r, d = d, λ = λ)

	#Seperate both data frames into testing data and training data
    train_data = subset(data, :time => time -> time .<= T1)
    test_data = subset(data, :time => time -> time .> T1)

	#Generate NODE model 
    model = NODE(train_data);

	#Train the model. If speedy, cut off training early. 
    if (speedy)
        gradient_descent!(model, maxiter = 50)
    else 
        gradient_descent!(model)
    end
    return model, test_data
end

#=Generates an array of initial conditions and then plots the model's predictions for each set of initial conditions
in a phase plane. 
=#
function phase_plane(model; start = .05, stop = 1.00, step = .1, max_T = 250)
    u0_array = [] 
    for u01 in range(start, stop = stop, step = step)
        for u02 in range(start, stop = stop, step = .1)
            if (u01 + u02 > 1) #can't have total percent area be greater than one
                continue 
            end
            push!(u0_array, [u01, u02]) 
        end 
    end 
    return UniversalDiffEq.phase_plane(model, u0_array, T = 250)
end 

function state_estimates(model)
    return UniversalDiffEq.plot_state_estimates(model)
end 

