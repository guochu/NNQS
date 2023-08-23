module NNQS

# there are 5 important imgredients for Neural Network Quantum States:
# 1) The Neural Network Quantum States ansatz, which accept a computational basis and returns its amplitude
# 2) The Markov Chain sampling algorithm
# 3) a way to randomly change the computational basis, which may reflect the symmetry of the model
# 4) The variational Monte Carlo algorithm, the essense of which is to evaluate the gradients.
# 5) The Hamiltonian representation which allows to efficiently derive coupled states given an computational basis

using Random, LinearAlgebra, Statistics
using Zygote, Flux
using Zygote: Params, Grads


# # auxiliary functions
# export parameters, reset!

# neural network states
export AbstractNNQS, Ψ, FCN, RBM
export MPS, rightorth!, rightorth, isrightcanonical

# sampler
export BitFlip, BondSwap, FermiBondSwap, move!
export AbstractSampler, MetropolisLocal, Metropolis, thermalize!, update!, init_state, generate_samples
export AutoRegressiveSampler, autoregressivesampling

# hamiltonian
export Hamiltonian, coupled_states, diagonal_coupling
export EnergiesGrads, BlockEnergiesGrads, aggregate
export energy, energy_and_grad, energy_and_grad_sr
# the exact versions are used for debug
export energy_exact, energy_and_grad_exact, energy_and_grad_sr_exact


# auxiliary
# NN module rewrites layers since Flux only support Float32 currently, while we need Float64 and ComplexF64
include("auxiliary/nn/basic.jl")

using NNQS.NN

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

end