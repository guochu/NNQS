"""
	energy_and_grad(h, nnqs, sampler; kwargs...)
	Add a nonzero regularization to the loss for nonzero λ
"""
function energy_and_grad(h::Hamiltonian, nnqs::AbstractNNQS, sampler::AbstractSampler; n_chain::Int=10, 
	seeds::Union{Vector{Int}, Nothing}=nothing, λ::Real = 1.0e-6, verbosity::Int=1)
	energies_and_grads = energy_and_grad_per_rank(h, nnqs, sampler, n_chain=n_chain, seeds=seeds)
	energy, grad = energies_and_grads[1]
	energies = zeros(typeof(energy), length(energies_and_grads))
	energies[1] = energy
	for i in 2:length(energies_and_grads)
		a, b = energies_and_grads[i]
		grad .+= b
		energies[i] = a
	end
	E_loc = mean(energies)
	grad ./= length(energies_and_grads)

	if verbosity >= 1
		println("Ē = $E_loc ± $(std(energies))")
	end

	return E_loc, _regularization!(grad, Flux.destructure(nnqs)[1], λ)
end

function energy_and_grad_per_rank(h::Hamiltonian, nnqs::AbstractNNQS, sampler::AbstractSampler; n_chain::Int=10, 
	seeds::Union{Vector{Int}, Nothing}=nothing)

	function f_per_chain(_seed)
		Random.seed!(_seed)
		_energy, grad = energy_and_grad_per_chain(h, nnqs, sampler)
		grad, re = Flux.destructure(grad)
		return _energy, grad
	end
	if isnothing(seeds)
		seeds = rand(1000:100000, n_chain)
	else
		@assert length(seeds) == n_chain
	end
	
	return f_per_chain.(seeds)
end

function energy_and_grad_per_chain(h::Hamiltonian, nnqs::AbstractNNQS, sampler::AbstractSampler)
	mean_energy, back = Zygote.pullback(_energy, h, nnqs, sampler)
	_a, grad, _b = back(one(mean_energy))
	@assert isnothing(_a) && isnothing(_b)
	return mean_energy, grad
end

function energy(h::Hamiltonian, nnqs::AbstractNNQS, sampler::AbstractSampler; n_chain::Int=10, seeds::Union{Vector{Int}, Nothing}=nothing, verbosity::Int=1)
	function f_per_chain(_seed)
		Random.seed!(_seed)
		_E = _energy(h, nnqs, sampler)
		return _E
	end
	if isnothing(seeds)
		seeds = rand(1000:100000, n_chain)
	else
		@assert length(seeds) == n_chain
	end
	
	energies = f_per_chain.(seeds)
	E_loc = mean(energies)
	if verbosity >= 1
		println("Ē = $E_loc ± $(std(energies))")
	end
	return E_loc
end

function _energy(h::Hamiltonian, nnqs::AbstractNNQS, sampler::AbstractSampler)
	unique_samples, counts = sampling(h, nnqs, sampler)
	amps = Ψ(nnqs, unique_samples)
	_energies = local_energies(h, nnqs, unique_samples, amps)
	weights = counts ./ sum(counts)
	return real(dot(weights, _energies))
end

Zygote.@adjoint _energy(h::Hamiltonian, nnqs::AbstractNNQS, sampler::AbstractSampler) = begin
	unique_samples, counts = sampling(h, nnqs, sampler)
	amps, amps_back = Zygote.pullback(Ψ, nnqs, unique_samples)
	_energies = local_energies(h, nnqs, unique_samples, amps)
	weights = counts ./ sum(counts)
	mean_energy = dot(weights, _energies)
	_energies .-= mean_energy
	return real(mean_energy), z -> begin
		weighted_energies = weights .* _energies
		_loss, _loss_back = Zygote.pullback(_energy_util, amps, weighted_energies)
		_loss_grad, _n = _loss_back(2*one(_loss))
		@assert isnothing(_n)
		amps_grad, _n = amps_back(_loss_grad)
		@assert isnothing(_n)
		return nothing, amps_grad, nothing
	end
end



function log_amplitudes(amps)
	if eltype(amps) <: Real
		return log.(abs.(amps))
	else
		return log.(amps)
	end
end

_energy_util(amps, weighted_energies) = dot(log_amplitudes(amps), dropgrad(weighted_energies))


