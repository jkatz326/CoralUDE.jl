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

