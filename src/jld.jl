using JLD2, UniversalDiffEq

#=Save a the right hand side of a trained model to a file using JLD2. 
https://jack-h-buckner.github.io/UniversalDiffEq.jl/dev/modelanalysis/
=#

function save_rhs(file_name, model;is_path = false)
    rhs = UniversalDiffEq.get_right_hand_side(model) #Get RHS 
    if (is_path) #Interpret file_name as path,
        JLD2.save_object(file_name, rhs) #Save to this path
    else #Else, interpret file_name as name
        path = file_name * ".jld2" #Append file format and save locally 
        JLD2.save_object(path, rhs)
    end 
end

#Load right hand side of previously trained model from a file using JLD2
function load_rhs(file_name;is_path = false)
    if (is_path)
        rhs = JLD2.load_object(file_name) #Load rhs from path 
    else 
        path = file_name * ".jld2" #Load from name 
        rhs = JLD2.load_object(path)
    end 
    return rhs
end 

function combine_data_ude_inputs(dn, un)
    return (maxiter = un.maxiter, seed = dn.seed, datasize = dn.datasize, σ1 = dn.σ1, σ2 = dn.σ2, T1 = un.T1, T2 = dn.T, u01 = dn.u01, u02 = dn.u02, a = dn.a, γ = dn.γ, r = dn.r, d = dn.d, λ = dn.λ)
end 

#Save a trained model using JLD2
function save_model(file_name, model, inputs;is_path = false)
    parameters = model.parameters #Get parameters 
    to_save = (inputs, parameters) #Combine inputs and calibrated parameters 
    if (is_path) #Interpret file_name as path,
        JLD2.save_object(file_name, to_save) #Save to this path
    else #Else, interpret file_name as name
        path = file_name * ".jld2" #Append file format and save locally 
        JLD2.save_object(path, to_save)
    end 
end 

function load_model(file_name;is_path = false)
    if (is_path)
        saved = JLD2.load_object(file_name) #Load rhs from path 
    else 
        path = file_name * ".jld2" #Load from name 
        saved = JLD2.load_object(path)
    end 
    in, parameters = saved 
    return ude_model(maxiter = in.maxiter, seed = in.seed, datasize = in.datasize, σ1 = in.σ1, σ2 = in.σ2, T1 = in.T1, T2 = in.T2, u01 = in.u01, u02 = in.u02, a = in.a, γ = in.γ, r = in.r, d = in.d, λ = in.λ, return_inputs = false, saved_parameters = parameters)
end


