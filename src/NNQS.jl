module NNQS

# there are 5 important imgredients for Neural Network Quantum States:
# 1) The Neural Network Quantum States ansatz, which accept a computational basis and returns its amplitude
# 2) The Markov Chain sampling algorithm
# 3) a way to randomly change the computational basis, which may reflect the symmetry of the model
# 4) The variational Monte Carlo algorithm, the essense of which is to evaluate the gradients.
# 5) The Hamiltonian representation which allows to efficiently derive coupled states given an computational basis

# # auxiliary functions
# export parameters, reset!

# neural network states
export AbstractNNQS, Ψ, FCN, RBM
export MPS, rightorth!, rightorth, isrightcanonical, increase_bond!

# sampler
export BitFlip, BondSwap, FermiBondSwap, move!
# Metropolis-Hasting sampling
export AbstractSampler, MetropolisLocal, Metropolis, thermalize!, update!, init_state
# auto-regressive sampling
export AutoRegressiveSampler, autoregressivesampling
# batchautoregressivesampling with various constrains
export BatchAutoRegressiveSampler, batchautoregressivesampling
export AbstractSymmetryConstrain, satisfied
export NoConstrain, MaskUnphysical, DiscardUnphysical
export U1Discard, U1Mask, U1U1Discard, U1U1Mask

# hamiltonian
export Hamiltonian, coupled_states, diagonal_coupling
export energy, energy_and_grad, energy_and_grad_sr, sampling
# the exact versions are used for debug
export energy_exact, energy_and_grad_exact, energy_and_grad_sr_exact

# utilities
export IsingChain, Ising2D, HeisenbergChain

using Random, LinearAlgebra, Statistics, Distributions
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
include("sampler/constrains.jl")
include("sampler/autoregressive.jl")
include("sampler/batchautoregressive.jl")

# hamiltonian 
include("hamiltonian/hamiltonian.jl")

# utility
include("utility/spin_hamiltonians.jl")

end