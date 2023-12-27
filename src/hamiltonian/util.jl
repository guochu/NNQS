
function _regularization!(grad, θs, λ)
	if λ != zero(λ)
		@. grad += (2 * λ) * conj(θs)
	end	
	return grad
end


"""
	generate_samples(h::Hamiltonian, m::Metropolis, nnqs::AbstractNNQS)
Generate a fixed number of samples using MC sampling
"""
function generate_samples(h::Hamiltonian, m::Metropolis, nnqs::AbstractNNQS)
	state = init_state(h, m.N, m.mover)
	work_state = copy(state)
	thermalize!(m, nnqs, state, work_state)
	samples = zeros(Int, m.N, m.n_sample_per_chain)
	for i in 1:m.n_sample_per_chain
		samples[:, i] = _update!(m, nnqs, state, work_state)
	end
	return unique_counts(samples)
end
function generate_samples(h::Hamiltonian, m::AutoRegressiveSampler, nnqs::AbstractNNQS)
	@assert m.N == sys_size(nnqs)
	samples = zeros(Int, m.N, m.n_sample_per_chain)
	nnqs2 = rightorth(nnqs, normalize=true)
	for i in 1:m.n_sample_per_chain
		samples[:, i] = autoregressivesampling(nnqs2)
	end	
	return unique_counts(samples)
end
function generate_samples(h::Hamiltonian, m::BatchAutoRegressiveSampler, nnqs::AbstractNNQS)
	@assert m.N == sys_size(nnqs)
	nnqs2 = rightorth(nnqs, normalize=true)
	return batchautoregressivesampling(nnqs2, m.n_sample_per_chain, m.constrain)
end
sampling(h::Hamiltonian, nnqs::AbstractNNQS, m::AbstractSampler) = generate_samples(h, m, nnqs)

function _count_map(samples::BatchComputationBasis)
	@assert !isempty(samples)
	sample_conuts = Dict{Vector{Int}, Int}()
	for i in 1:size(samples, 2)
		sample = samples[:, i]
		c = get(sample_conuts, sample, 0)
		sample_conuts[sample] = c + 1
	end
	return sample_conuts
end
function _count_map(samples::AbstractVector{<:ComputationBasis})
	@assert !isempty(samples)
	sample_conuts = Dict{Vector{Int}, Int}()
	for sample in samples
		c = get(sample_conuts, sample, 0)
		sample_conuts[sample] = c + 1
	end
	return sample_conuts
end

function unique_counts(samples)
	sample_conuts = _count_map(samples)
	L = length(sample_conuts)
	N = size(samples, 1)
	unique_samples = zeros(Int, N, L)
	counts = zeros(Int, L)
	i = 1
	for (k, v) in sample_conuts
		unique_samples[:, i] = k
		counts[i] = v
		i += 1
	end
	return unique_samples, counts
end

# backpropagation of each sample
function local_energy_and_grad(h::Hamiltonian, nnqs::AbstractNNQS, state::ComputationBasis)
	# amp_state, back = Zygote.pullback( () -> Ψ(nnqs, state), Flux.params(nnqs) )
	amp_state, back = Zygote.pullback( m -> Ψ(m, state), nnqs )
	_energy = local_energy(h, nnqs, state, amp_state)
	grad, re = Flux.destructure(back(one(_energy)))
	grad ./= conj(amp_state)
	return _energy, grad
end

# converting -1 and 1 into 2 and 1
index_to_state(index) = [(index[i] == 2) ? -1 : 1 for i in 1:length(index)]
state_to_index(state) = CartesianIndex(Tuple((item == -1) ? 2 : 1 for item in state))

function compute_E∂θ(energies::Vector{<:Number}, θs::Vector{<:Vector}, weights::Vector{<:Real})
	@assert length(energies) == length(θs) == length(weights)
	n_sample = length(energies)
	Eθs_ave = zero(θs[1])
	for i in 1:n_sample
		# Eθs_ave .+= energies[i] .* view(θs, :, i) 
		axpy!(energies[i] * weights[i], θs[i], Eθs_ave)
	end
	return Eθs_ave
end
function compute_∂θ∂θ(∂θ::Vector{<:Vector}, weights::Vector{<:Real})
	@assert length(∂θ) == length(weights)
	N = length(∂θ[1])
	∂θ∂θ = zeros(eltype(∂θ[1]), N * N)
	n_sample = length(∂θ)
	for i in 1:n_sample
		∂θ_batch = ∂θ[i]
		# ∂θ∂θ .+= kron(conj(∂θ_batch), ∂θ_batch)
		_inplace_kron!(∂θ∂θ, ∂θ_batch, weights[i])
	end	
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