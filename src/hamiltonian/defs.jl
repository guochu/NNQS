
abstract type Hamiltonian end


Base.eltype(h::Hamiltonian) = error("eltype not implemented for NNQS type $(typeof(h)).")

"""
	coupled_states(h::Hamiltonian, state::ComputationBasis)
	return a list of states together with coefficients

	There is the only interface for customized Hamiltonian, one could also implement
	diagonal_coupling for best efficiency
"""
coupled_states(h::Hamiltonian, state::ComputationBasis) = error("coupled_states not implemented for hamiltonian type $(typeof(h)).")

"""
	diagonal_coupling(h::Hamiltonian, state::ComputationBasis)
	return the coefficient of the diagonal term of the given computational basis.
	One could also implemented a general coupled_state which includes the diagonal
	terms, in which case this function is not necessary. But it can be split off
	from coupled_states for efficiency.
"""
diagonal_coupling(h::Hamiltonian, state::ComputationBasis) = error("diagonal_coupling not implemented for hamiltonian type $(typeof(h)).")

"""
	init_state(h::Hamiltonian, N::Int, mover::StateChangeRule)
	generate an initial state for MC sampling
"""
init_state(h::Hamiltonian, N::Int, mover::StateChangeRule) = _init_state(N, mover)

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
		samples[:, i] = update!(m, nnqs, state, work_state)
	end
	return unique_counts(samples)
end
function generate_samples(h::Hamiltonian, m::AutoRegressiveSampler, nnqs::MPS)
	@assert m.N == sys_size(nnqs)
	samples = zeros(Int, m.N, m.n_sample_per_chain)
	nnqs2 = rightorth(nnqs, normalize=true)
	for i in 1:m.n_sample_per_chain
		samples[:, i] = autoregressivesampling(nnqs2)
	end	
	return unique_counts(samples)
end
function generate_samples(h::Hamiltonian, m::BatchAutoRegressiveSampler, nnqs::MPS)
	@assert m.N == sys_size(nnqs)
	nnqs2 = rightorth(nnqs, normalize=true)
	return batchautoregressivesampling(nnqs2, m.n_sample_per_chain, m.constrain)
end

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

