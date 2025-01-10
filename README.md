# CoralUDE

[![Build Status](https://github.com/jkatz326/CoralUDE.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/jkatz326/CoralUDE.jl/actions/workflows/CI.yml?query=branch%3Amaster)

For use in upcoming paper by Dr. Emerson Arehart. Requires Julia 1.11.

To install and load CoralUDE, open Julia and type the following code:
```
] add https://github.com/jkatz326/CoralUDE.jl.git
```

No official documentation at this stage. To learn more, navigate to src -> mumby_model.jl and read comments.

# Tutorial

As a simple tutorial, run the following code:
```
using CoralUDE
model, test_data = CoralUDE.ude_model()
CoralUDE.state_estimates(model)
```
This should generate the following plot:
![Screenshot 2025-01-09 210643](https://github.com/user-attachments/assets/3690edf0-1155-451d-a228-f5b0630e950f)
