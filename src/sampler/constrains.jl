abstract type AbstractConstrain end

satisfied(c::AbstractConstrain, state::ComputationBasis) = error("satisfied for constrain type $(typeof(c))")


struct NoConservation <: AbstractConstrain end
satisfied(c::NoConservation, state::ComputationBasis) = true


# conserve Nsamples on each node
abstract type NodeConservationConstrain <: AbstractConstrain end
# conserve Nsamples in the end
abstract type RootConservationConstrain <: AbstractConstrain end
# conserve Nsamples only at the beginning
abstract type LeafConservationConstrain <: AbstractConstrain end

struct U1LeafConservation <: LeafConservationConstrain
	N::Int
	c::Int

function U1LeafConservation(N::Int, c::Int)
	@assert N > 0
	@assert 0 <= c <= N
	new(N, c)
end

end
U1LeafConservation(N::Int; c::Int) = U1LeafConservation(N, c)


struct U1NodeConservation <: NodeConservationConstrain
	N::Int
	c::Int

function U1NodeConservation(N::Int, c::Int)
	@assert N > 0
	@assert 0 <= c <= N
	new(N, c)
end

end
U1NodeConservation(N::Int; c::Int) = U1NodeConservation(N, c)


satisfied(x::Union{U1LeafConservation, U1NodeConservation}, state::ComputationBasis) = x.c-(x.N-length(state)) <= sum_positive(state) <= x.c


struct U1U1LeafConservation <: LeafConservationConstrain
	N::Int
	c₁::Int
	c₂::Int

function U1U1LeafConservation(N::Int, c₁::Int, c₂::Int) 
	@assert (N > 0) && (N % 2 == 0)
	@assert (0 <= c₁ <= div(N, 2)) && (0 <= c₂ <= div(N, 2))
	new(N, c₁, c₂)
end

end
U1U1LeafConservation(N::Int; c₁::Int, c₂::Int=c₁) = U1U1LeafConservation(N, c₁, c₂)


struct U1U1NodeConservation <: NodeConservationConstrain
	N::Int
	c₁::Int
	c₂::Int

function U1U1NodeConservation(N::Int, c₁::Int, c₂::Int) 
	@assert (N > 0) && (N % 2 == 0)
	@assert (0 <= c₁ <= div(N, 2)) && (0 <= c₂ <= div(N, 2))
	new(N, c₁, c₂)
end

end
U1U1NodeConservation(N::Int; c₁::Int, c₂::Int=c₁) = U1U1NodeConservation(N, c₁, c₂)

function satisfied(x::Union{U1U1LeafConservation, U1U1NodeConservation}, state::ComputationBasis)
	L = length(state)
	return isodd(L) ? (x.c₁-(x.N-L) <= sum_positive(view(state, 1:2:L)) <= x.c₁) : (x.c₂-(x.N-L) <= sum_positive(view(state, 2:2:L)) <= x.c₂)
end

function sum_positive(state::AbstractVector{Int})
	r = 0
	for item in state
		if item == 1
			r += 1
		end
	end
	return r
end

