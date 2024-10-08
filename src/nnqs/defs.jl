
# abstract definition of neural network quantum state

abstract type AbstractNNQS end
const ComputationBasis = AbstractVector{Int}
const BatchComputationBasis = AbstractMatrix{Int}
const GeneralBasis = AbstractVecOrMat{Int}

Base.eltype(::Type{T}) where {T<:AbstractNNQS} = error("eltype not implemented for NNQS type $(T)")
Base.eltype(nnqs::AbstractNNQS) = eltype(typeof(nnqs))

Ψ(nnqs::AbstractNNQS, state::GeneralBasis, args...) = _Ψ(nnqs, dropgrad(state), args...)


# interfaces that should be implemented by a customized NNQS model

"""
	_Ψ(nnqs::AbstractNNQS, state)
	if state is a specific computational basis, return the amplitude of it
	if state is a batch of computational basis, return the amplitudes of them as a vector. 
	The batched version should be explicitly implemented for better efficiency.
	One must also implemented the correct adjoint function of Ψ, which should return the gradients
	as a vector.

	Interfaces one must implemented for customized Neural Network Quantum States:
	1) Ψ 
	2) Base.eltype: used to deduce the element type of the parameters
	3) nparameters: return total number of parameters
	4) parameters: return all the parameters
	5) reset_parameters!: reset all the parameters in the NNQS with the given parameters
	
	In the future I may consider use a machine learning framework such as Flux as the backend, then one would 
	be able to use the parameters deduction mechanism of Flux and 3,4,5) will no longer be required.
"""
_Ψ(nnqs::AbstractNNQS, state::ComputationBasis, args...) = error("_Ψ not defined for NNQS type $(typeof(nnqs)).")
_Ψ(nnqs::AbstractNNQS, state::BatchComputationBasis, args...) = transpose([_Ψ(nnqs, view(state, :, j), args...) for j in 1:size(state, 2)])

# used for exact sampler
sys_size(nnqs::AbstractNNQS) = error("sys_size not implemented for NNQS type $(typeof(nnqs)).")

# this is required for better efficiency
# Base.eltype(nnqs::AbstractNNQS) = error("eltype not implemented for NNQS type $(typeof(nnqs)).")
