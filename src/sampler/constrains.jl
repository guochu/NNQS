abstract type AbstractSymmetryConstrain end

satisfied(c::AbstractSymmetryConstrain, state::ComputationBasis) = error("satisfied for constrain type $(typeof(c))")


struct NoConstrain <: AbstractSymmetryConstrain end
satisfied(c::NoConstrain, state::ComputationBasis) = true


# conserve Nsamples on each node
abstract type MaskUnphysical <: AbstractSymmetryConstrain end
# conserve Nsamples only at the beginning
abstract type DiscardUnphysical <: AbstractSymmetryConstrain end
# # conserve Nsamples in the end
# abstract type RootConservationConstrain <: AbstractSymmetryConstrain end

struct U1Discard <: DiscardUnphysical
	N::Int
	c::Int

function U1Discard(N::Int, c::Int)
	@assert N > 0
	@assert 0 <= c <= N
	new(N, c)
end

end
U1Discard(N::Int; c::Int) = U1Discard(N, c)


struct U1Mask <: MaskUnphysical
	N::Int
	c::Int

function U1Mask(N::Int, c::Int)
	@assert N > 0
	@assert 0 <= c <= N
	new(N, c)
end

end
U1Mask(N::Int; c::Int) = U1Mask(N, c)


satisfied(x::Union{U1Discard, U1Mask}, state::ComputationBasis) = x.c-(x.N-length(state)) <= sum_positive(state) <= x.c


struct U1U1Discard <: DiscardUnphysical
	N::Int
	c₁::Int
	c₂::Int

function U1U1Discard(N::Int, c₁::Int, c₂::Int) 
	@assert (N > 0) && (N % 2 == 0)
	@assert (0 <= c₁ <= div(N, 2)) && (0 <= c₂ <= div(N, 2))
	new(N, c₁, c₂)
end

end
U1U1Discard(N::Int; c₁::Int, c₂::Int=c₁) = U1U1Discard(N, c₁, c₂)


struct U1U1Mask <: MaskUnphysical
	N::Int
	c₁::Int
	c₂::Int

function U1U1Mask(N::Int, c₁::Int, c₂::Int) 
	@assert (N > 0) && (N % 2 == 0)
	@assert (0 <= c₁ <= div(N, 2)) && (0 <= c₂ <= div(N, 2))
	new(N, c₁, c₂)
end

end
U1U1Mask(N::Int; c₁::Int, c₂::Int=c₁) = U1U1Mask(N, c₁, c₂)

function satisfied(x::Union{U1U1Discard, U1U1Mask}, state::ComputationBasis)
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

