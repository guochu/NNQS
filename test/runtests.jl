push!(LOAD_PATH, "../src")

include("util.jl")


using Test
using Random, Zygote
using NNQS, Flux

function test_rbm_grad_single(::Type{T}) where T
	n_visible = 2
	nqs = RBM(T, n_hidden=4, n_visible=n_visible)
	state = rand(-1:2:1, n_visible)

	loss_origin(nn) = real(Ψ(nn, state))

	loss(p) = begin
		reset!(Flux.params(nqs), p)
		return loss_origin(nqs)
	end

	grad1 = parameters(gradient(() -> loss_origin(nqs), Flux.params(nqs)))
	grad2 = fdm_gradient(loss, parameters(Flux.params(nqs)))

	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function test_rbm_grad_batch(::Type{T}) where T
	n_hidden = 30
	n_visible = 5
	n_batch = 100
	nqs = RBM(T, n_hidden=n_hidden, n_visible=n_visible)
	state = rand(-1:2:1, n_visible, n_batch)

	loss_origin(nn) = real(sum(Ψ(nn, state)))

	loss(p) = begin
		reset!(Flux.params(nqs), p)
		return loss_origin(nqs)
	end

	grad1 = parameters(gradient(() -> loss_origin(nqs), Flux.params(nqs)))
	grad2 = fdm_gradient(loss, parameters(Flux.params(nqs)))

	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_local_energy_grad(::Type{T}, L::Int, h::Real, J::Real) where T
	n_visible = L
	nqs = RBM(T, n_hidden=2*n_visible, n_visible=n_visible)

	ham = NNQS.IsingChain(h=h, J=J)

	loss_origin(nn) = real(energy_exact(ham, nn))

	loss(p) = begin
		reset!(Flux.params(nqs), p)
		return loss_origin(nqs)
	end

	E, grad1 = energy_and_grad_exact(ham, nqs)

	grad2 = fdm_gradient(loss, parameters(Flux.params(nqs)))

	# println("energy is $E")

	# println(grad1)
	# println(grad2)

	return maximum(abs.(grad1 - grad2)) < 1.0e-6
	# return grad1, grad2
end


println("Testing neural network gradient...")

@testset "Testing RBM gradients" begin
    @test test_rbm_grad_single(Float64)
    @test test_rbm_grad_single(ComplexF64)
    @test test_rbm_grad_batch(Float64)
    @test test_rbm_grad_batch(ComplexF64)

    @test check_local_energy_grad(Float64, 4, 1, 1)
    @test check_local_energy_grad(ComplexF64, 4, -1, 1)
    @test check_local_energy_grad(ComplexF64, 5, 1, 1)
end
