abstract type AbstractConstrain end

struct NoConstrain <: AbstractConstrain end

struct U1Constrain <: AbstractConstrain
	n::Int
end
U1Constrain(; n::Int) = U1Constrain(n)

struct U1U1Constrain <: AbstractConstrain
	n₁::Int
	n₂::Int
end
U1U1Constrain(; n₁::Int, n₂::Int) = U1U1Constrain(n₁, n₂)

