println("------------------------------------")
println("|         MPS gradients            |")
println("------------------------------------")

@testset "Checking MPS gradients" begin
	tol = 1.0e-5
	
	for T in (Float64, ComplexF64)
		n_visible = 2
		nqs = MPS(T, n_visible, D=3)
		state = rand(-1:2:1, n_visible)
		x0, re = Flux.destructure(nqs)

		loss_origin(nn) = real(Ψ(nn, state))
		loss(p) = loss_origin(re(p))


		out1 = gradient(nn -> loss_origin(nn), nqs)
		grad1 = Flux.destructure(out1)[1]
		grad2 = fdm_gradient(loss, x0)
		@test maximum(abs.(grad1 - grad2)) < tol
	end

	for T in (Float64, ComplexF64)
		n_hidden = 30
		n_visible = 5
		n_batch = 100
		nqs = MPS(T, n_visible, D=4)
		state = rand(-1:2:1, n_visible, n_batch)
		x0, re = Flux.destructure(nqs)

		loss_origin(nn) = real(sum(Ψ(nn, state)))
		loss(p) = loss_origin(re(p))

		out1 = gradient(nn -> loss_origin(nn), nqs)
		grad1 = Flux.destructure(out1)[1]
		grad2 = fdm_gradient(loss, x0)

		@test maximum(abs.(grad1 - grad2)) < tol
	end

	for T in (Float64, ComplexF64)
		for L in (4, 5)
			for h in (1, -1)
				J = 1
				n_visible = L
				nqs = MPS(T, n_visible, D=5)
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