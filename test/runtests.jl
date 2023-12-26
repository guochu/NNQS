

include("util.jl")


using Test
using Random, Zygote, Flux
using LinearAlgebra

push!(LOAD_PATH, "../src")
using NNQS

# include("../src/includes.jl")

include("gradients.jl")
