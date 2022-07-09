
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

