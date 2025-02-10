include("mumby_model.jl")
model, test_data = ude_model(σ1 = 0, σ2 = 0)
phase_plane(model, step = .05)