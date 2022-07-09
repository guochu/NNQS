abstract type StateChangeRule end


struct BitFlip <: StateChangeRule end
struct BondSwap <: StateChangeRule 
	charge::Int
end
BondSwap(;charge::Int=0) = BondSwap(charge)

"""
	struct FermiBondSwap <: StateChangeRule
"""
struct FermiBondSwap <: StateChangeRule
	charge::Tuple{Int, Int}
end
FermiBondSwap(;charge::Tuple{Int, Int}=(0, 0)) = FermiBondSwap(charge)


"""
	move!(state::ComputationBasis, sc::StateChangeRule) 
	a StateChangeRule changes the current computational basis. One could implement symmetries such 
	as number conservation by implementing customized StateChangeRule.

	The functionn move! is the only interface function needed for a customized StateChangeRule.
"""
move!(state::ComputationBasis, sc::StateChangeRule) = error("move! not implemented for StateChangeRule type $(typeof(sc)).")

function move!(state::ComputationBasis, sc::BitFlip)
	site = rand(1:length(state))
	state[site] = -state[site]
	return state
end

"""
	init_state(N::Int, mover)
	return a random initial state
	This is an interface function which should be separately implemented for each StateChangeRule to incorperate symmetry
"""
init_state(N::Int, mover::BitFlip) = rand(-1:2:1, N)

function move!(state::ComputationBasis, sc::BondSwap)
	site = rand(1:length(state)-1)
	state[site], state[site+1] = state[site+1], state[site]
	return state
end
function init_state(L::Int, mover::BondSwap)
	charge = mover.charge
	m_charge = _neural_charge(charge, L)
	sites = _get_charged_sites(m_charge, L)
	state = [-1 for i in 1:L]
	for site in sites
		state[site] = 1
	end
	return state
end

function move!(state::ComputationBasis, sc::FermiBondSwap)
	@assert iseven(length(state))
	L = div(length(state), 2)
	site_up = rand(1:L-1)

	# if rand() < 0.5
	# 	state[2*site_up-1], state[2*site_up+1] = state[2*site_up+1], state[2*site_up-1]
	# else
	# 	state[2*site_up], state[2*site_up+2] = state[2*site_up+2], state[2*site_up]
	# end
	j = rand(1:3)
	if j == 1
		state[2*site_up-1], state[2*site_up+1] = state[2*site_up+1], state[2*site_up-1]
	elseif j == 2
		state[2*site_up], state[2*site_up+2] = state[2*site_up+2], state[2*site_up]
	else
		state[2*site_up-1], state[2*site_up+1] = state[2*site_up+1], state[2*site_up-1]
		state[2*site_up], state[2*site_up+2] = state[2*site_up+2], state[2*site_up]
	end

	
	# state[2*site_up], state[2*site_up+2] = state[2*site_up+2], state[2*site_up]

	# site_down = rand(1:L-1)
	# state[2*site_down], state[2*site_down+2] = state[2*site_down+2], state[2*site_down]
	return state
end
function init_state(N::Int, mover::FermiBondSwap)
	@assert iseven(N)
	charge = mover.charge
	L = div(N, 2)
	m_charge_up = _neural_charge(charge[1], L)
	m_charge_down = _neural_charge(charge[2], L)
	sites_up = _get_charged_sites(m_charge_up, L)
	sites_down = _get_charged_sites(m_charge_down, L)
	state = [-1 for i in 1:N]
	for site in sites_up
		state[2*site - 1] = 1
	end
	for site in sites_down
		state[2*site] = 1
	end
	return state
end

function _neural_charge(charge::Int, L::Int)
	@assert (charge >= -L) && (charge <= L)
	((charge + L) % 2 == 0) || throw(ArgumentError("wrong charge"))
	m_charge = div(charge + L, 2)
	return m_charge
end
function _get_charged_sites(m_charge::Int, L::Int)
	sites = Set{Int}()
	while length(sites) != m_charge
		push!(sites, rand(1:L))
	end	
	@assert length(sites) == m_charge
	return sites
end
