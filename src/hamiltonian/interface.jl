
abstract type Hamiltonian end


Base.eltype(h::Hamiltonian) = error("eltype not implemented for NNQS type $(typeof(h))")

"""
	coupled_states(h::Hamiltonian, state::ComputationBasis)
	return a list of states together with coefficients

One could either implement coupled_states or local_energy
"""
coupled_states(h::Hamiltonian, state::ComputationBasis) = error("coupled_states not implemented for hamiltonian type $(typeof(h))")

"""
	diagonal_coupling(h::Hamiltonian, state::ComputationBasis)

return the coefficient of the diagonal term of the given computational basis.
One could also implemented a general coupled_state which includes the diagonal
terms, in which case this function is not necessary. But it can be split off
from coupled_states for efficiency.
"""
diagonal_coupling(h::Hamiltonian, state::ComputationBasis) = zero(eltype(h))

"""
	init_state(h::Hamiltonian, N::Int, mover::StateChangeRule)
Generate an initial state for MC sampling, not used for autoregressive sampling
"""
init_state(h::Hamiltonian, N::Int, mover::StateChangeRule) = _init_state(N, mover)

"""
	local_energy(h::Hamiltonian, nnqs::AbstractNNQS, state::ComputationBasis)
	return E_local
"""
function local_energy(h::Hamiltonian, nnqs::AbstractNNQS, state::ComputationBasis, amp_state::Number=Ψ(nnqs, state))
	E_loc = diagonal_coupling(h, state)
	c_states, coefs = coupled_states(h, state)
	if !isempty(coefs)
		E_loc += only((Ψ(nnqs, c_states) ./ amp_state) * coefs)
	end
	return E_loc
end

function local_energies(h::Hamiltonian, nnqs::AbstractNNQS, state::BatchComputationBasis, amps=Ψ(nnqs, state))
	@assert size(state, 2) == length(amps)
	return [local_energy(h, nnqs, view(state, :, j), amps[j]) for j in 1:length(amps)]
end