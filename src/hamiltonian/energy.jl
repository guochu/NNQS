

function energy(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS)
	samples = generate_samples(h, sampler, nnqs)
	_energies = [local_energy_single(h, nnqs, view(samples, :, j)) for j in 1:size(samples, 2)]
	return mean(_energies)
end

function energy_and_grad_util_per_rank(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; n_chain_per_rank::Int=10, 
	seeds::Union{Vector{Int}, Nothing}=nothing)

	function f_per_chain(_seed)
		Random.seed!(_seed)
		return energies_and_grads_one_chain(h, nnqs, sampler)
	end
	if isnothing(seeds)
		seeds = rand(1000:100000, n_chain_per_rank)
	else
		@assert length(seeds) == n_chain_per_rank
	end
	
	return BlockEnergiesGrads(f_per_chain.(seeds))
end

"""
	energy_and_grad(h, sampler, nnqs; kwargs...)
	Add a nonzero regularization to the loss for nonzero λ
	"""
function energy_and_grad(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; n_chain_per_rank::Int=10, 
	seeds::Union{Vector{Int}, Nothing}=nothing, λ::Real = 1.0e-6, verbosity::Int=1)
	# samples = generate_samples(sampler, nnqs)
	block_energies_∂θs = energy_and_grad_util_per_rank(h, sampler, nnqs, n_chain_per_rank=n_chain_per_rank, seeds=seeds)
	E_loc, grad = compute_energy_and_grad(aggregate(block_energies_∂θs), parameters(Flux.params(nnqs)), λ=λ)

	if verbosity >= 1
		mean_energies = [mean(block_energies_∂θs[i].energies) for i in 1:n_chain_per_rank]
		println("Ē = $E_loc ± $(std(mean_energies))")
	end

	return E_loc, grad
end

function energy_and_grad_sr(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; n_chain_per_rank::Int=10, 
	seeds::Union{Vector{Int}, Nothing}=nothing, diag_shift::Real=1.0e-2, λ::Real = 1.0e-6, verbosity::Int=1)
	# samples = generate_samples(sampler, nnqs)
	block_energies_∂θs = energy_and_grad_util_per_rank(h, sampler, nnqs, n_chain_per_rank=n_chain_per_rank, seeds=seeds)
	E_loc, grad = compute_energy_and_grad_sr(aggregate(block_energies_∂θs), parameters(Flux.params(nnqs)), diag_shift=diag_shift, λ=λ)

	if verbosity >= 1
		mean_energies = [mean(block_energies_∂θs[i].energies) for i in 1:n_chain_per_rank]
		println("Ē = $E_loc ± $(std(mean_energies))")
	end

 	# # regularization
 	# S .+= reg .* Diagonal(S)
 	# # grad_sr = S \ grad
 	# grad_sr = pinv(S) * grad
 	# return E_loc, grad_sr

 	# return E_loc, stable_solve(S, grad, diag_shift=diag_shift)
	return E_loc, grad
end




# function energy(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS)
# 	samples = generate_samples!(sampler, nnqs, thermalize=true)
# 	_energies = [local_energy_single(h, nnqs, view(samples, :, j)) for j in 1:size(samples, 2)]
# 	return mean(_energies)
# end

# function energy_and_grad(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; λ::Real = 1.0e-6, verbosity::Int=1)
# 	samples = generate_samples!(sampler, nnqs, thermalize=true)
# 	E_loc, ∂θ = local_energy_and_grad(h, nnqs, view(samples, :, 1))

# 	E_loc = zero(E_loc)
# 	∂θ = zero(∂θ)
# 	E∂θ = zero(∂θ)
# 	n_sample = size(samples, 2)
# 	_energies = eltype(E_loc)[]

# 	for n in 1:n_sample
# 		# println("states are $(sampler.state)")
# 		E_loc_batch, ∂θ_batch = local_energy_and_grad(h, nnqs, view(samples, :, n))
# 		E_loc += E_loc_batch
# 		@. ∂θ += ∂θ_batch
# 		@. E∂θ += E_loc_batch * ∂θ_batch
# 		push!(_energies, E_loc_batch)
# 	end
# 	E_loc /= n_sample
# 	∂θ ./= n_sample
# 	E∂θ ./= n_sample

# 	if verbosity >= 1
# 		@printf("Ē = %f ± %f\n", E_loc, std(_energies))
# 	end

# 	return E_loc, _regularization!(2 .* (E∂θ .- real(E_loc) .* ∂θ), parameters(Flux.params(nnqs)), λ)
# end

# function energy_and_grad_sr(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; diag_shift::Real=1.0e-2, λ::Real = 1.0e-6, verbosity::Int=1)
# 	samples = generate_samples!(sampler, nnqs, thermalize=true)
# 	E_loc, ∂θ = local_energy_and_grad(h, nnqs, view(samples, :, 1))

# 	E_loc = zero(E_loc)
# 	∂θ = zero(∂θ)
# 	E∂θ = zero(∂θ)
# 	∂θ∂θ = zeros(eltype(∂θ), length(∂θ), length(∂θ))
# 	n_sample = size(samples, 2)

# 	_energies = eltype(E_loc)[]
# 	for n in 1:n_sample
# 		E_loc_batch, ∂θ_batch = local_energy_and_grad(h, nnqs, view(samples, :, n))
# 		E_loc += E_loc_batch
# 		@. ∂θ += ∂θ_batch
# 		@. E∂θ += E_loc_batch * ∂θ_batch
# 		∂θ∂θ .+= reshape(kron(conj(∂θ_batch), ∂θ_batch), length(∂θ), length(∂θ))
# 		push!(_energies, E_loc_batch)
# 	end
# 	E_loc /= n_sample
# 	∂θ ./= n_sample
# 	E∂θ ./= n_sample
# 	∂θ∂θ ./= n_sample
# 	grad = 2 .* (E∂θ .- real(E_loc) .* ∂θ)
# 	S = ∂θ∂θ - reshape(kron(conj(∂θ), ∂θ), length(∂θ), length(∂θ))

# 	if verbosity >= 1
# 		@printf("Ē = %f ± %f\n", E_loc, std(_energies))
# 	end


#  	# # regularization
#  	# S .+= reg .* Diagonal(S)
#  	# # grad_sr = S \ grad
#  	# grad_sr = pinv(S) * grad
#  	# return E_loc, grad_sr

#  	# return E_loc, stable_solve(S, grad, diag_shift=diag_shift)
# 	return E_loc, _regularization!(stable_solve(S, grad, diag_shift=diag_shift), parameters(Flux.params(nnqs)), λ)
# end







