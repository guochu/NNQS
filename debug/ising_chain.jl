push!(LOAD_PATH, "../src")

using Random
using NNQS
using Flux, Flux.Optimise


function test_gs_energy(L; h, J)
	ham = NNQS.IsingChain(h=h, J=J)
	n_hidden = 2 * L
	n_visible = L
	Random.seed!(1234)
	rbm = FCN(Float64, n_hidden=n_hidden, n_visible=n_visible)
	sampler = MetropolisLocal(n_visible, n_thermal=10000, n_sample_per_chain=500, n_discard=100)	


	# gradient descent algorithm using ADAM 

	learn_rate = 0.01
	epoches = 200
	opt = ADAM(learn_rate)

	x0, re = Flux.destructure(rbm)

	losses = Float64[]

    for i in 1:epoches
        train_loss, grad = energy_and_grad(ham, rbm, sampler, n_chain=10, λ=1.0e-5)

        Optimise.update!(opt, x0, grad)
        rbm = re(x0)

        push!(losses, train_loss)
        println("energy at the $i-th step is $(train_loss).")
    end
    return losses
end


function test_gs_energy2(L; h, J)
	ham = NNQS.IsingChain(h=h, J=J)
	n_hidden = 2 * L
	n_visible = L
	Random.seed!(1234)
	rbm = MPS(Float64, n_visible, D=4)
	sampler = MetropolisLocal(n_visible, n_thermal=10000, n_sample_per_chain=500, n_discard=100)	


	# gradient descent algorithm using ADAM 

	learn_rate = 0.01
	epoches = 200
	opt = ADAM(learn_rate)

	x0, re = Flux.destructure(rbm)

	losses = Float64[]

    for i in 1:epoches
        train_loss, grad = energy_and_grad(ham, rbm, sampler, n_chain=10, λ=1.0e-5)

        Optimise.update!(opt, x0, grad)
        rbm = re(x0)

        push!(losses, train_loss)
        println("energy at the $i-th step is $(train_loss).")
    end
    return losses
end

function test_gs_energy3(L; h, J)
	ham = NNQS.IsingChain(h=h, J=J)
	n_hidden = 2 * L
	n_visible = L
	Random.seed!(1234)
	rbm = MPS(Float64, n_visible, D=4)
	rightorth!(rbm)
	sampler = BatchAutoRegressiveSampler(n_visible, n_sample_per_chain=50000)	


	# gradient descent algorithm using ADAM 

	learn_rate = 0.005
	epoches = 500
	opt = ADAM(learn_rate)

	x0, re = Flux.destructure(rbm)

	losses = Float64[]

    for i in 1:epoches
        train_loss, grad = energy_and_grad(ham, rbm, sampler, n_chain=10, λ=1.0e-5)

        Optimise.update!(opt, x0, grad)
        rbm = re(x0)

        push!(losses, train_loss)
        println("energy at the $i-th step is $(train_loss).")
    end
    return losses
end
