println("------------------------------------")
println("|         RBM gradients            |")
println("------------------------------------")

# function test_rbm_grad_single(::Type{T}) where T
# 	n_visible = 2
# 	nqs = RBM(T, n_hidden=4, n_visible=n_visible)
# 	state = rand(-1:2:1, n_visible)

# 	loss_origin(nn) = real(Ψ(nn, state))

# 	loss(p) = begin
# 		reset!(Flux.params(nqs), p)
# 		return loss_origin(nqs)
# 	end

# 	grad1 = parameters(gradient(() -> loss_origin(nqs), Flux.params(nqs)))
# 	grad2 = fdm_gradient(loss, parameters(Flux.params(nqs)))

# 	return maximum(abs.(grad1 - grad2)) < 1.0e-6
# end

# function test_rbm_grad_batch(::Type{T}) where T
# 	n_hidden = 30
# 	n_visible = 5
# 	n_batch = 100
# 	nqs = RBM(T, n_hidden=n_hidden, n_visible=n_visible)
# 	state = rand(-1:2:1, n_visible, n_batch)

# 	loss_origin(nn) = real(sum(Ψ(nn, state)))

# 	loss(p) = begin
# 		reset!(Flux.params(nqs), p)
# 		return loss_origin(nqs)
# 	end

# 	grad1 = parameters(gradient(() -> loss_origin(nqs), Flux.params(nqs)))
# 	grad2 = fdm_gradient(loss, parameters(Flux.params(nqs)))

# 	return maximum(abs.(grad1 - grad2)) < 1.0e-6
# end

# function check_local_energy_grad(::Type{T}, L::Int, h::Real, J::Real) where T
# 	n_visible = L
# 	nqs = RBM(T, n_hidden=2*n_visible, n_visible=n_visible)

# 	ham = NNQS.IsingChain(h=h, J=J)

# 	loss_origin(nn) = real(energy_exact(ham, nn))

# 	loss(p) = begin
# 		reset!(Flux.params(nqs), p)
# 		return loss_origin(nqs)
# 	end

# 	E, grad1 = energy_and_grad_exact(ham, nqs)

# 	grad2 = fdm_gradient(loss, parameters(Flux.params(nqs)))

# 	# println("energy is $E")

# 	# println(grad1)
# 	# println(grad2)

# 	return maximum(abs.(grad1 - grad2)) < 1.0e-6
# 	# return grad1, grad2
# end

# function check_local_energy_grad_mps(::Type{T}, L::Int, h::Real, J::Real) where T
# 	nqs = MPS(T, L, D=3)

# 	ham = NNQS.IsingChain(h=h, J=J)

# 	loss_origin(nn) = real(energy_exact(ham, nn))

# 	loss(p) = begin
# 		reset!(Flux.params(nqs), p)
# 		return loss_origin(nqs)
# 	end

# 	E, grad1 = energy_and_grad_exact(ham, nqs)

# 	grad2 = fdm_gradient(loss, parameters(Flux.params(nqs)))

# 	# println("energy is $E")

# 	# println(grad1)
# 	# println(grad2)

# 	return maximum(abs.(grad1 - grad2)) < 1.0e-6
# 	# return grad1, grad2
# end


@testset "Checking RBM gradients" begin
	tol = 1.0e-5

	for T in (Float64, ComplexF64)
		n_visible = 2
		nqs = RBM(T, n_hidden=4, n_visible=n_visible)
		models = [RBM(T, n_hidden=4, n_visible=n_visible), MPS(T, n_visible, D=3)]
		state = rand(-1:2:1, n_visible)

		for nqs in models
			x0, re = Flux.destructure(nqs)
			loss_origin(nn) = real(Ψ(nn, state))
			loss(p) = loss_origin(re(p))


			out1 = gradient(nn -> loss_origin(nn), nqs)
			grad1 = Flux.destructure(out1)[1]
			grad2 = fdm_gradient(loss, x0)

			@test maximum(abs.(grad1 - grad2)) < tol
		end
	end

	for T in (Float64, ComplexF64)
		n_hidden = 30
		n_visible = 5
		n_batch = 100
		models = [RBM(T, n_hidden=n_hidden, n_visible=n_visible), MPS(T, n_visible, D=4)]
		state = rand(-1:2:1, n_visible, n_batch)
		for nqs in models
			x0, re = Flux.destructure(nqs)
			loss_origin(nn) = real(sum(Ψ(nn, state)))
			loss(p) = loss_origin(re(p))

			out1 = gradient(nn -> loss_origin(nn), nqs)
			grad1 = Flux.destructure(out1)[1]
			grad2 = fdm_gradient(loss, x0)

			@test maximum(abs.(grad1 - grad2)) < tol
		end
	end

	for T in (Float64, ComplexF64)
		for L in (4, 5)
			for h in (1, -1)
				J = 1
				n_visible = L
				models = [RBM(T, n_hidden=2*n_visible, n_visible=n_visible), MPS(T, n_visible, D=5)]

				for nqs in models
					x0, re = Flux.destructure(nqs)
					ham = NNQS.IsingChain(h=h, J=J)
					loss_origin(nn) = real(energy_exact(ham, nn))
					loss(p) = loss_origin(re(p))
					E, grad1 = energy_and_grad_exact(ham, nqs)
					grad2 = fdm_gradient(loss, x0)

					@test maximum(abs.(grad1 - grad2)) < tol
				end
			end
		end
	end
end
