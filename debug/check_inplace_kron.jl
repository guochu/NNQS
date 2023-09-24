# push!(LOAD_PATH, "src")

# using NeuralNetworkStates, BenchmarkTools
using LinearAlgebra

function _inplace_kron!(∂θ∂θ, ∂θ, scale)
	N = length(∂θ)
	@inbounds for i in 1:N
		axpy!(scale * conj(∂θ[i]), ∂θ, view(∂θ∂θ, (i-1)*N+1:i*N))
	end
	return ∂θ∂θ
end

function check_kron(N)
	A1 = randn(ComplexF64, N * N)
	A2 = copy(A1)

	v1 = randn(ComplexF64, N)
	v2 = copy(v1)

	println("1*************************************")
	@time A1 .+= kron(conj(v1), v1)

	println("2*************************************")
	@time _inplace_kron!(A2, v2, 1)

	return maximum(abs.(A1 - A2))
end