push!(LOAD_PATH, "../src")

include("util.jl")


using Test
using Random, Zygote
using NNQS, Flux
using LinearAlgebra

include("gradients.jl")
# include("mps.jl")