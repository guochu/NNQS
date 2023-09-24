println("------------------------------------")
println("|          NN gradients            |")
println("------------------------------------")


@testset "Checking NN gradients" begin
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

@testset "Checking batchautoregressive energy and gradient" begin
	tol = 1.0e-3

	for T in (Float64, )
		for L in (4, 5)
			sampler = BatchAutoRegressiveSampler(L, n_sample_per_chain=10^10)
			for h in (1, -1)
				J = 1
				n_visible = L
				nqs = MPS(T, n_visible, D=5)
				ham = NNQS.IsingChain(h=h, J=J)
				
				E1, grad1 = energy_and_grad_exact(ham, nqs)
				E2, grad2 = energy_and_grad(ham, nqs, sampler, verbosity=0)

				@test abs((E1-E2)/ E1) < tol
				@test norm(grad1 - grad2) / norm(grad1) < tol
			end
		end
	end	
end

