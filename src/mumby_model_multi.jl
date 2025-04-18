using UniversalDiffEq, Lux, DataFrames, StochasticDiffEq, OrdinaryDiffEq, Distributions, Random, Plots, DiffEqFlux

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
function valid_range2(v)
	return max(min(1 - 1e-2, v), 1e-2)
end;

# Smooth maximum: approximates max(x, y) with a differentiable function.
function smooth_max(x, y; α=50.0)
    return log(exp(α*x) + exp(α*y)) / α
end

# Smooth minimum: approximates min(x, y) with a differentiable function.
function smooth_min(x, y; α=50.0)
    return -log(exp(-α*x) + exp(-α*y)) / α
end

# Smooth clamp: first smooth-min with the upper bound, then smooth-max with the lower bound.
function valid_range(v; lower=1e-2, upper=1-1e-2, α=50.0)
    return smooth_max(lower, smooth_min(v, upper; α=α); α=α)
end

function valid_range2(v)
    return clamp(v, 1e-2, 1 - 1e-2)
end

scaled_tanh(x) = 0.5 * (tanh(x) + 1)

function softmax(u)
    xe = 1-sum(u)
    sm = exp(xe)+sum(exp.(u))
    return exp.(u) ./ sm
end 

function inv_softmax(p)
    p = p.+10^-6
    pe = 1-sum(p)
    c = (1-sum(log.(p))-log(pe))/(length(p)+1)
    return log.(p) .+ c 
end 

function convert_to_UDE(custom_ude)::UDE
    # Create a new UDE instance by copying all the fields from CustomUDE
    return UDE(
        custom_ude.times,
        custom_ude.data,
        custom_ude.X,
        custom_ude.data_frame,
        custom_ude.X_data_frame,
        custom_ude.parameters,
        custom_ude.loss_function,
        custom_ude.process_model,
        custom_ude.process_loss,
        custom_ude.observation_model,
        custom_ude.observation_loss,
        custom_ude.process_regularization,
        custom_ude.observation_regularization,
        custom_ude.constructor,
        custom_ude.time_column_name,
        custom_ude.weights,
        custom_ude.variable_column_name,
        custom_ude.value_column_name,
        (ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    )
end

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
    u1, u2 = u
	a, γ, r, d, λ, _ = p
    ϵ = .01
    g = λ(t)
    du[1] = a*u1*u2 - (g*u1)/(1 - u2 + ϵ) + γ*u1*(1 - u1 - u2)
    du[2] = r*(1 - u1 - u2)*u2 - d*u2 - a*u1*u2 + .01
end

#Process Noise:
#dw function for system of SDEs, additive process noise for grazing term 
function mumby_noise(du, u, p, t)
    u1, u2 = u
    _, _, _, _, _, σ1 = p
    ϵ = .01
    du[1] = σ1*u1/(1 - u2 + ϵ)
    du[2] = 0
end

#Observation Noise, add comments

function add_obs_noise(u, σ2)
    u_tr = inv_softmax(u)
    u_tr = u_tr .+ rand(Normal(0.0, σ2), length(u_tr))
    return softmax(u_tr)
end


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
Generates a data set of simulated coral reef dynamics using the ODEs above. Used to train the Universal Differential Equation model specified below. 
Parameters-- 
- plot : true generates a plot of the data
- seed : random seed for data Generation
- datasize : number of data points at which we save the output of the ODE solver 
- σ1 : parameter that controls process noise
- σ2 : parameter that controls observation noise
- T : data will be generated from time 0 to time T
- The remaining paramters are from the Mumby model. See above. 
=#
function coral_data(;plot = false, seed = 123, datasize = 60, σ1 = 0, σ2 = 0, T = 300, u01 = .2, u02 = .2, a = .1, γ = .8, r = 1, d = .44, λ = constant_fun(.3), return_inputs = false)
    # set seed 
    Random.seed!(seed)

    # set parameters
    tspan = (0.0f0, T)
    tsteps = range(tspan[1], tspan[2], length = datasize)

    # model parameters
    u0 = Float32[u01, u02]
    p = (a = a, γ = γ, r = r, d = d, λ = λ, σ1 = σ1)

    # generate time series with DifferentialEquations.jl
    prob_trueode = SDEProblem(mumby_eqns, mumby_noise, u0, tspan, p)
    solution = solve(prob_trueode, SOSRI(), saveat = tsteps, save_noise=true)
    ode_data = Array(solution)

    noise_data = solution.W.(solution.t)
    noise_applied = [noise[1][1] for noise in noise_data]
    effective_g = λ.(solution.t) .+ (noise_applied .* σ1)


    #Add Observation Noise
    ode_data = min.(1 - 1e-6, max.(ode_data, 1e-6))
    ode_data = hcat([add_obs_noise(ode_data[:, i], σ2) for i in 1:size(ode_data, 2)]...)

    #format data, return now if only data requested
    data = DataFrame(x1 = ode_data[1,:], x2 = ode_data[2,:], time = tsteps)
    if !(plot || return_inputs)
        return data 
    end 

    # collect inputs if requested
    inputs = nothing
    if return_inputs
        inputs = (seed = seed, datasize = datasize, T = T, σ1 = σ1, σ2 = σ2, u01 = u01, u02 = u02, a = a, γ = γ, r = r, d = d, λ = λ)
    end 

    # generate plot if requested, return 
    if plot
        plt = Plots.scatter(tsteps,transpose(ode_data), xlabel = "Time", ylabel = ["Macroalgae % Cover", "Coral % Cover"], label = "")
        if return_inputs
            return data, plt, inputs 
        end
        return data, plt 
    end
    return data, inputs 
end

function merge_coral_data(df1, df2)
    newdf1 = copy(df1)
    newdf2 = copy(df2)
    newdf1[!, :series] .= 1
    newdf2[!, :series] .= 2
    combined = vcat(newdf1, newdf2)
    return combined
end 

#Universal Differential Equation Model:
#=
Using the ODEs described above, generates a model of the equations where the death rate term is replaced by a neural network. 
This neural network is trained on a subset of the synthetic data that is designated for training. 
Then, the function outputs the trained model along with the remaining data for testing.
Parameters-- 
- maxiter : max number of iterations for gradient descent. Default is negative which does not set a max number of iterations. 
- Training data will be generated from times 0 to T1
- Testing data will be generated from times T1 + 1 to last time in coral data 
- Description of other model parameters can be found in code block above. 
=#

function ude_model_from_data(data;maxiter = -1, T1 = 175, a = .1, γ = .8, r = 1, d = .44, return_inputs = false, saved_parameters = nothing)
	
    #Seperate both data frames into testing data and training data 
    train_data = subset(data, :time => time -> time .<= T1)
    test_data = subset(data, :time => time -> time .> T1)

    # set neural network dimensions
    num_series = 2
    dims_out = 1
    dims_in = 3 + num_series
    hidden = 5

    # Define neural network with Lux.jl
    NN = Lux.Chain(Lux.Dense(dims_in, hidden, tanh), Lux.Dense(hidden, dims_out))
    rng = Random.default_rng()
    NNparameters, NNstates = Lux.setup(rng,NN)
    parameters = (NN = NNparameters, a = a, γ = γ, r = r, d = d)
	
    # Define derivatives (time dependent NODE)
    function derivs!(du, u, i, p, t)
        index = round(Int, i)
        one_hot = zeros(num_series, 1)
        one_hot[index] = 1
        u1, u2 = scaled_tanh.(u)
        vals = scaled_tanh.(NN(vcat([u1, u2, t], one_hot), p.NN, NNstates)[1])
        a, γ, r, d = p.a, p.γ, p.r, p.d
        du[1] = a * u1 * u2 + vals[1] + γ * u1 * (1 - u1 - u2)
        du[2] = r * (1 - u1 - u2) * u2 - d * u2 - a * u1 * u2
    end

	#Generate model using UniversalDiffEq
    model = MultiCustomDerivatives(train_data,derivs!,parameters;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)
    
    #state_variable_transform = x -> 1 ./ (1 .+ exp.(-x)
	#Train the model. Use saved paramters if available (see jld.jl). Otherwise, train using gradient descent. Stop after maxiter steps if specified. 
    if !isnothing(saved_parameters)
        model.parameters = saved_parameters
    elseif (maxiter < 0)
        gradient_descent!(model)
    else 
        gradient_descent!(model, maxiter = maxiter)
    end

    #If returning inputs, collect inputs and output with model and test data. Otherwise, output output model and test data.
    if (return_inputs)
        inputs = (maxiter = maxiter, T1 = T1, a = a, γ = γ, r = r, d = d)
        return model, test_data, inputs 
    else 
        return model, test_data
    end 
end

#Wrapper around ude_model_from_data that generates the data before training the model.
function ude_model(;maxiter = -1, seed = 123, datasize = 60, σ1 = 0, σ2 = 0, T1 = 175, T2 = 300, u01 = .2, u02 = .2, a = .1, γ = .8, r = 1, d = .44, λ = constant_fun(.3), return_inputs = false, saved_parameters = nothing)
	
    #Generate Synthetic data using Mumby Equations
    data = coral_data(plot = false, seed = seed, datasize = datasize, σ1 = σ1, σ2 = σ2, u01 = u01, u02 = u02, a = a, γ = γ, r = r, d = d, λ = λ)

    #Generate the model 
    model, test_data = ude_model_from_data(data, maxiter = maxiter, T1 = T1, a = a, γ = γ, r = r, d = d, return_inputs = false, saved_parameters = saved_parameters)

    #If returning inputs, collect inputs and output with model and test data. Otherwise, output model and test data. 
    if (return_inputs)
        inputs = (maxiter = maxiter, seed = seed, datasize = datasize, σ1 = σ1, σ2 = σ2, T1 = T1, T2 = T2, u01 = u01, u02 = u02, a = a, γ = γ, r = r, d = d, λ = λ)
        return model, test_data, inputs
    else 
        return model, test_data
    end 
end

#Same as the UDE function above but trains a generic neural net [NODE] rather than a UDE
function node_model(data;maxiter = -1, seed = 123, datasize = 60, σ1 = 0, σ2 = 0, T1 = 175, T2 = 300, u01 = .2, u02 = .2, a = .1, γ = .8, r = 1, d = .44, λ = constant_fun(.3), return_inputs = false, saved_parameters = nothing)

	#Seperate both data frames into testing data and training data
    train_data = subset(data, :time => time -> time .<= T1)
    test_data = subset(data, :time => time -> time .> T1)

	#Generate NODE model 
    model = MultiNODE(train_data)

	#Train the model. Use saved paramters if available (see jld.jl). Otherwise, train using gradient descent. Stop after maxiter steps if specified. 
    if !isnothing(saved_parameters)
        model.parameters = saved_parameters
    elseif (maxiter < 0)
        gradient_descent!(model)
    else 
        gradient_descent!(model, maxiter = maxiter)
    end
    
    #If returning inputs, collect inputs and output with model and test data. Otherwise, output output model and test data. 
    if (return_inputs)
        inputs = (maxiter = maxiter, seed = seed, datasize = datasize, σ1 = σ1, σ2 = σ2, T1 = T1, T2 = T2, u01 = u01, u02 = u02, a = a, γ = γ, r = r, d = d, λ = λ)
        return model, test_data, inputs
    else 
        return model, test_data
    end 
end

#=Generates an array of initial conditions and then plots the model's predictions for each set of initial conditions
in a phase plane. 
=#
function phase_plane(model; start = .05, stop = 1.00, step = .1, max_T = 250, title = "Plot Title", xlabel = "Macroalgae Cover", ylabel = "Coral Cover")
    
    #Generate array of initial conditions to plot forecast for 
    u0_array = [] 
    for u01 in range(start, stop = stop, step = step)
        for u02 in range(start, stop = stop, step = step)
            if (u01 + u02 > 1) #can't have total percent area be greater than one
                continue 
            end
            push!(u0_array, [u01, u02]) 
        end 
    end 

    #Generate plot and add input labels 
    plt = UniversalDiffEq.phase_plane(model, u0_array, T = max_T)
    title!(plt, title)
    xlabel!(plt, xlabel)
    ylabel!(plt, ylabel)
    return plt 
end 

function state_estimates(model)
    return UniversalDiffEq.plot_state_estimates(model)
end 

