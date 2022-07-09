module NN

using Flux: @functor, identity

"""	
	struct Dense{A <: AbstractMatrix, B <: AbstractVector, F <: Function}
	I rewrite Dense because in Flux it assumes Float32 scalar type, but we want Float64 or even ComplexF64
"""
struct Dense{A <: AbstractMatrix, B <: AbstractVector, F <: Function}
	W::A
	b::B
	σ::F
end

(x::Dense)(v::AbstractVecOrMat) = (x.σ).(x.W * v .+ x.b)

function Dense(::Type{T}, idim::Integer, odim::Integer, σ::Function=identity) where {T<:Number}
	W = randn(T, odim, idim)
	b = randn(T, odim)
	return Dense(W, b, σ)
end

Base.copy(x::Dense) = Dense(copy(x.W), copy(x.b), x.σ)

@functor Dense



end