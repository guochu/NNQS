# restricted Boltzmann machine, see G. Carleo and M. Troyer, Science 355, 602 (2017) for details

# Flux export the same logcosh function 
log_cosh(x::Number) = log(cosh(x))

struct FCN{D, A<:AbstractVector, F <: Function} <: AbstractNNQS
	dn::D
	a::A
	activation::F
end

function FCN(::Type{T}; n_visible::Int, n_hidden::Int, activation::Function=Flux.identity) where {T <: Number}
	sca = convert(real(T), 0.01) 
	a = randn(T, n_visible) .* sca
	b = randn(T, n_hidden) .* sca
	W = randn(T, n_hidden, n_visible) .* sca
	m = NN.Dense(W, b, log_cosh)
	return FCN(m, a, activation)
end
Base.eltype(::Type{FCN{D, A, F}}) where {D, A, F} = eltype(A)

Flux.@functor FCN

function Base.getproperty(m::FCN, x::Symbol)
	if x == :b
		return m.dn.b
	elseif x == :W
		return m.dn.W
	elseif x == :σ
		return m.dn.σ
	elseif x == :n_hidden
		return size(m.dn.W, 1)
	elseif x == :n_visible
		return size(m.dn.W, 2)
	else
		return getfield(m, x)
	end
end

sys_size(m::FCN) = m.n_visible
Base.copy(m::FCN) = FCN(copy(m.dn), copy(m.a), m.activation)


_Ψ(m::FCN, x::ComputationBasis) = m.activation(transpose(m.a) * x) * exp(sum(m.dn(x) ))
# the output is a 1×batch_size array
_Ψ(m::FCN, x::BatchComputationBasis) = (m.activation).(transpose(m.a) * x) .* exp.(sum( m.dn(x), dims=1))



RBM(::Type{T}; kwargs...) where {T <: Number} = FCN(T; activation=exp, kwargs...)

# struct RBM{D, A<:AbstractVector} <: AbstractNNQS
# 	dn::D
# 	a::A
# end

# function RBM(::Type{T}; n_visible::Int, n_hidden::Int) where {T <: Number}
# 	sca = convert(real(T), 0.01) 
# 	a = randn(T, n_visible) .* sca
# 	b = randn(T, n_hidden) .* sca
# 	W = randn(T, n_hidden, n_visible) .* sca
# 	m = NN.Dense(W, b, logcosh)
# 	return RBM(m, a)
# end

# Flux.@functor RBM

# function Base.getproperty(m::RBM, x::Symbol)
# 	if x == :b
# 		return m.dn.b
# 	elseif x == :W
# 		return m.dn.W
# 	elseif x == :σ
# 		return m.dn.σ
# 	elseif x == :n_hidden
# 		return size(m.dn.W, 1)
# 	elseif x == :n_visible
# 		return size(m.dn.W, 2)
# 	else
# 		return getfield(m, x)
# 	end
# end

# sys_size(m::RBM) = m.n_visible
# Base.copy(m::RBM) = RBM(copy(m.dn), copy(m.a))


# Ψ(m::RBM, x::ComputationBasis) = exp(sum(vcat(transpose(m.a) * x, m.dn(x) )))
# Ψ(m::RBM, x::BatchComputationBasis) = exp.(sum(vcat(transpose(m.a) * x, m.dn(x) ), dims=1))


