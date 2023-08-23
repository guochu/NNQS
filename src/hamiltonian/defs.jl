
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
	return samples
end
function generate_samples(h::Hamiltonian, m::AutoRegressiveSampler, nnqs::MPS)
	@assert m.N == sys_size(nnqs)
	samples = zeros(Int, m.N, m.n_sample_per_chain)
	nnqs2 = rightorth(nnqs, normalize=true)
	for i in 1:m.n_sample_per_chain
		samples[:, i] = autoregressivesampling(nnqs2)
	end	
	return samples
end

