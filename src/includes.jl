
using Random, LinearAlgebra, Statistics
using Zygote, Flux
using Zygote: Params, Grads

# auxiliary
# NN module rewrites layers since Flux only support Float32 currently, while we need Float64 and ComplexF64
include("auxiliary/nn/basic.jl")

using .NN

# params.jl provides functions to convert Params object from and to a flat vector of parameters, this is
# necessary when we want to do stochastic reconfiguration
include("auxiliary/params.jl")

# neural network states
include("nnqs/defs.jl")
include("nnqs/rbm.jl")
include("nnqs/mps.jl")

# sampler
include("sampler/defs.jl")
include("sampler/mover.jl")
include("sampler/Metropolis.jl")
include("sampler/autoregressive.jl")

# hamiltonian 
include("hamiltonian/defs.jl")

include("hamiltonian/energy_util.jl")
include("hamiltonian/energy.jl")
# the exact calculations are used for debug
include("hamiltonian/energy_exact.jl")


# utility
include("utility/spin_hamiltonians.jl")
