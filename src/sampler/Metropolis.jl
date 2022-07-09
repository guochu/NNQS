
struct Metropolis{M <: StateChangeRule} <: AbstractSampler
	N::Int
	n_thermal::Int
	n_sample_per_chain::Int
	n_discard::Int
	# state::Vector{Int}
	# work_state::Vector{Int}
	mover::M
end

Metropolis(N::Int; mover::StateChangeRule, n_thermal::Int=10000, n_sample_per_chain::Int=500, n_discard::Int=100) = Metropolis(
	N, n_thermal, n_sample_per_chain, n_discard, mover)
MetropolisLocal(N::Int; kwargs...) = Metropolis(N; mover=BitFlip(), kwargs...)


# """
# 	init!(m::Metropolis)
# 	initialize the internal state. 
# 	This is an interface function which should be separately implemented for each StateChangeRule to incorperate symmetry
# """
# function init!(m::Metropolis{BitFlip})
# 	for i in 1:length(m.state)
# 		m.state[i] = rand(-1:2:1)
# 	end	
# end

# function init!(m::Metropolis{BondSwap})
# 	charge = m.mover.charge
# 	L = length(m.state)
# 	m_charge = _neural_charge(charge, L)
# 	sites = _get_charged_sites(m_charge, L)
# 	m.state .= -1
# 	for site in sites
# 		m.state[site] = 1
# 	end
# end
# function init!(m::Metropolis{FermiBondSwap})
# 	charge = m.mover.charge
# 	L = div(length(m.state), 2)
# 	m_charge_up = _neural_charge(charge[1], L)
# 	m_charge_down = _neural_charge(charge[2], L)
# 	sites_up = _get_charged_sites(m_charge_up, L)
# 	sites_down = _get_charged_sites(m_charge_down, L)
# 	m.state .= -1
# 	for site in sites_up
# 		m.state[2*site - 1] = 1
# 	end
# 	for site in sites_down
# 		m.state[2*site] = 1
# 	end
# end

# function thermalize!(m::AbstractSampler, nnqs::AbstractNNQS)
# 	init!(m)
# 	for i in 1:m.n_thermal
# 		single_update!(m, nnqs)
# 	end
# end

# function single_update!(m::Metropolis, nnqs::AbstractNNQS)
# 	copyto!(m.work_state, m.state)
# 	move!(m.work_state, m.mover)
# 	p = abs2(Ψ(nnqs, m.work_state) / Ψ(nnqs, m.state))
# 	# println("trial state is $(m.work_state)")
# 	# println("p is $(p)")
# 	if rand() < min(1, p)
# 		copyto!(m.state, m.work_state)
# 	end
# 	return m.state
# end

function thermalize!(m::Metropolis, nnqs::AbstractNNQS, state::ComputationBasis, work_state::ComputationBasis)
	# init!(m)
	for i in 1:m.n_thermal
		single_update!(m, nnqs, state, work_state)
	end
	return state
end


function single_update!(m::Metropolis, nnqs::AbstractNNQS, state::ComputationBasis, work_state::ComputationBasis)
	copyto!(work_state, state)
	move!(work_state, m.mover)
	p = abs2(Ψ(nnqs, work_state) / Ψ(nnqs, state))
	if rand() < min(1, p)
		copyto!(state, work_state)
	end
	return state
end

function update!(sampler::Metropolis, nnqs::AbstractNNQS, state::ComputationBasis, work_state::ComputationBasis)
	for n in 1:sampler.n_discard
		single_update!(sampler, nnqs, state, work_state)
	end	
	return state
end

function generate_samples(m::Metropolis, nnqs::AbstractNNQS)
	state = init_state(m.N, m.mover)
	work_state = copy(state)
	thermalize!(m, nnqs, state, work_state)
	samples = zeros(Int, m.N, m.n_sample_per_chain)
	for i in 1:m.n_sample_per_chain
		samples[:, i] = update!(m, nnqs, state, work_state)
	end
	return samples
end



