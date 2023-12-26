
"""
	energy_and_grad_sr(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; kwargs...)
"""
function energy_and_grad_sr(h::Hamiltonian, nnqs::AbstractNNQS, sampler::AbstractSampler; n_chain::Int=10, 
	seeds::Union{Vector{Int}, Nothing}=nothing, diag_shift::Real=1.0e-2, λ::Real = 1.0e-6, verbosity::Int=1)
	# samples = generate_samples(sampler, nnqs)
	T = eltype(nnqs)
	counts = Int[]
	energies = Vector{T}()
	∂θs = Vector{Vector{T}}()
	if isnothing(seeds)
		seeds = rand(1000:100000, n_chain)
	else
		@assert length(seeds) == n_chain
	end
	# todo: regroup the unique samples and counts for efficiency
	for i in 1:n_chain
		Random.seed!(seeds[i])
		unique_samples_i, counts_i = sampling(h, nnqs, sampler)
		counts = append!(counts, counts_i)
		for j in 1:length(counts_i)
			_energy, _grad = local_energy_and_grad(h, nnqs, view(unique_samples_i, :, j))
			push!(energies, _energy)
			push!(∂θs, _grad)
		end
	end
	@assert length(counts) == length(energies) == length(∂θs)
	weights = counts ./ sum(counts)

	# compute the natural gradient
	return compute_energy_and_grad_sr(energies, ∂θs, weights, Flux.destructure(nnqs)[1], diag_shift=diag_shift, λ=λ)
end

function compute_energy_and_grad_sr(energies::Vector{<:Number}, θs::Vector{<:Vector}, weights::Vector{<:Real}, paras::Vector{<:Number}; diag_shift::Real=1.0e-2, λ::Real = 1.0e-6)
 	E_loc, grad, ∂θ = compute_energy_and_grad_util(energies, θs, weights)
	∂θ∂θ = compute_∂θ∂θ(θs, weights)
	S = reshape(_inplace_kron!(∂θ∂θ, ∂θ, -1), length(∂θ), length(∂θ))
	return E_loc, stable_solve(S, _regularization!(grad, paras, λ), diag_shift=diag_shift)
end

function _sum(θs::Vector{<:Vector}, weights::Vector{<:Real})
	@assert length(θs) == length(weights)
	m = zero(θs[1])
	for i in 1:length(θs)
		axpy!(weights[i], θs[i], m)
	end
	return m
end

function compute_energy_and_grad_util(energies::Vector{<:Number}, θs::Vector{<:Vector}, weights::Vector{<:Real})
	E_ave = dot(weights, energies)
	θs_ave = _sum(θs, weights)
	Eθs_ave = compute_E∂θ(energies, θs, weights)
	return E_ave, 2 .* (Eθs_ave .- real(E_ave) .* θs_ave), θs_ave
end