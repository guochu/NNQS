

struct EnergiesGrads{T<:Number, V<:AbstractVector}
	energies::Vector{T}
	grads::Vector{V}
end

struct BlockEnergiesGrads{E <: EnergiesGrads}
	data::Vector{E}
end

Base.append!(x::BlockEnergiesGrads, v::AbstractVector{<:EnergiesGrads}) = append!(x.data, v)
Base.append!(x::BlockEnergiesGrads, v::BlockEnergiesGrads) = append!(x.data, v.data)
Base.push!(x::BlockEnergiesGrads, v::EnergiesGrads) = push!(x.data, v)
Base.length(x::BlockEnergiesGrads) = length(x.data)
Base.getindex(x::BlockEnergiesGrads, i::Integer) = getindex(x.data, i)

function aggregate(m::BlockEnergiesGrads)
	r = m.data
	_energies = copy(r[1].energies)
	_grads = r[1].grads
	for i in 2:length(r)
		append!(_energies, r[i].energies)
		append!(_grads, r[i].grads)
	end
	return EnergiesGrads(_energies, _grads)
end


"""
	local_energy_single(h::Hamiltonian, nnqs::AbstractNNQS, state::ComputationBasis)
	return E_local
"""
function local_energy_single(h::Hamiltonian, nnqs::AbstractNNQS, state::ComputationBasis, amp_state::Number=Ψ(nnqs, state))
	E_loc = diagonal_coupling(h, state)
	c_states, coefs = coupled_states(h, state)
	if !isempty(coefs)
		E_loc += only((Ψ(nnqs, c_states) ./ amp_state) * coefs)
	end
	return E_loc
end

function local_energy_and_grad(h::Hamiltonian, nnqs::AbstractNNQS, state::ComputationBasis)
	amp_state, back = Zygote.pullback( () -> Ψ(nnqs, state), Flux.params(nnqs) )
	_energy = local_energy_single(h, nnqs, state, amp_state)
	grad = parameters(back(one(_energy)))
	grad ./= conj(amp_state)
	return _energy, grad
end

# function local_energy_and_grad_batch(h::Hamiltonian, nnqs::AbstractNNQS, state::BatchComputationBasis)
# 	r, back = Zygote.pullback(lnΨ, nnqs, state)
# 	energies = [local_energy_single(h, nnqs, state[:, j]) for j in 1:size(state, 2)]
# 	return sum(energies), back(ones(length(energies)))[1], back(energies)[1]
# end

function energies_and_grads_one_chain(h::Hamiltonian, nnqs::AbstractNNQS, samples::BatchComputationBasis)
	E_loc_single, ∂θ_single = local_energy_and_grad(h, nnqs, view(samples, :, 1))
	n_sample = size(samples, 2)
	energies = zeros(typeof(E_loc_single), n_sample)
	∂θs = Vector{typeof(∂θ_single)}(undef, n_sample) 
	for n in 1:n_sample
		energies[n], ∂θs[n] = local_energy_and_grad(h, nnqs, view(samples, :, n))
	end
	return EnergiesGrads(energies, ∂θs)
end
energies_and_grads_one_chain(h::Hamiltonian, nnqs::AbstractNNQS, sampler::AbstractSampler) = energies_and_grads_one_chain(
	h, nnqs, generate_samples(sampler, nnqs))

function _compute_energy_and_grad(energies::Vector{<:Number}, θs::Vector{<:Vector})
	E, grad, ∂θ = compute_energy_and_grad_util(energies, θs)
	return E, grad
end
compute_energy_and_grad(m::EnergiesGrads) = _compute_energy_and_grad(m.energies, m.grads)

function compute_energy_and_grad_util(energies::Vector{<:Number}, θs::Vector{<:Vector})
	n_sample = length(energies)
	E_ave = mean(energies)
	# θs_ave = mean(θs, dims=2)
	# θs_ave = reshape(θs_ave, length(θs_ave))
	θs_ave = _mean(θs)
	Eθs_ave = compute_E∂θ(energies, θs)
	return E_ave, 2 .* (Eθs_ave .- real(E_ave) .* θs_ave), θs_ave
end

function _mean(θs::Vector{<:Vector})
	m = zero(θs[1])
	for i in 1:length(θs)
		m .+= θs[i]
	end
	m ./= length(θs)
	return m
end

function _compute_energy_and_grad_sr(energies::Vector{<:Number}, θs::Vector{<:Vector}; diag_shift::Real=1.0e-2)
 	E_loc, grad, ∂θ = compute_energy_and_grad_util(energies, θs)
	∂θ∂θ = compute_∂θ∂θ(θs)
	# S = ∂θ∂θ - reshape(kron(conj(∂θ), ∂θ), length(∂θ), length(∂θ))
	S = reshape(_inplace_kron!(∂θ∂θ, ∂θ, -1), length(∂θ), length(∂θ))
	return E_loc, stable_solve(S, grad, diag_shift=diag_shift)
end
compute_energy_and_grad_sr(m::EnergiesGrads; kwargs...) = _compute_energy_and_grad_sr(m.energies, m.grads; kwargs...)

function compute_E∂θ(energies::Vector{<:Number}, θs::Vector{<:Vector})
	n_sample = length(energies)
	Eθs_ave = zero(θs[1])
	for i in 1:n_sample
		# Eθs_ave .+= energies[i] .* view(θs, :, i) 
		axpy!(energies[i], θs[i], Eθs_ave)
	end
	Eθs_ave ./= n_sample
	return Eθs_ave
end
function compute_∂θ∂θ(∂θ::Vector{<:Vector})
	N = length(∂θ[1])
	∂θ∂θ = zeros(eltype(∂θ[1]), N * N)
	n_sample = length(∂θ)
	for i in 1:n_sample
		∂θ_batch = ∂θ[i]
		# ∂θ∂θ .+= kron(conj(∂θ_batch), ∂θ_batch)
		_inplace_kron!(∂θ∂θ, ∂θ_batch, 1)
	end	
	∂θ∂θ ./= n_sample
	return ∂θ∂θ
end

function _inplace_kron!(∂θ∂θ, ∂θ, scale)
	N = length(∂θ)
	@inbounds for i in 1:N
		axpy!(scale * conj(∂θ[i]), ∂θ, view(∂θ∂θ, (i-1)*N+1:i*N))
	end
	return ∂θ∂θ
end

function stable_solve(S::AbstractMatrix, f::AbstractVector; diag_shift::Real=1.0e-2)
	# _svdvs = svdvals(Sm)
	# println("smallest singular value of S is $(_svdvs[end])")
	# println("conditional number of S is $(_svdvs[1] / _svdvs[end])")
	# S = copy(Sm)
	for i in 1:size(S, 1)
		S[i, i] += diag_shift
	end

	# Spc = similar(S)
	# for j in 1:size(S, 2)
	# 	for i in 1:size(S, 1)
	# 		Spc[i, j] = S[i, j] / (sqrt(S[i,i]) * sqrt(S[j, j]))
	# 	end
	# end
	# # for i in 1:size(S, 1)
	# # 	Spc[i, i] += diag_shift
	# # end

	# fpc = [f[i] / sqrt(S[i, i]) for i in 1:size(S, 1)]
	# # println("is there Nan? $(any(isnan.(fpc)))")


	# grad_sr = Spc \ fpc
	# return [grad_sr[i] / sqrt(S[i, i]) for i in 1:size(S, 1)]

	grad = S \ f
	return grad
end

function _regularization!(grad, θs, λ)
	if λ != zero(λ)
		@. grad += (2 * λ) * conj(θs)
	end	
	return grad
end
