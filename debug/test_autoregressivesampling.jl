push!(LOAD_PATH, "../src")
using NNQS

using Random, LinearAlgebra
using Flux, Flux.Optimise

function run_xxz_exact(L; h, J, Jzz)
    ham = NNQS.HeisenbergChain(h=h, J=J, Jzz=Jzz)
    m = NNQS.matrix(ham, L)
    evals = eigvals(Symmetric(m))
    return evals[1]
end


function main()

	L = 6
	h = 0.7
	J = 1
	Jzz = 1.3

	exact_energy = run_xxz_exact(L, h=h, J=J, Jzz=Jzz)
	println("exact energy is ", exact_energy)


	n_samples=10^10
	sampler_a = BatchAutoRegressiveSampler(L, n_sample_per_chain=n_samples)
	sampler_b = BatchAutoRegressiveSampler(L, n_sample_per_chain=n_samples, constrain=U1Discard(L, c=div(L,2)))
	sampler_c = BatchAutoRegressiveSampler(L, n_sample_per_chain=n_samples, constrain=U1Mask(L, c=div(L,2)))

	ham = NNQS.HeisenbergChain(h=h, J=J, Jzz=Jzz)
	learn_rate = 0.005
	epoches = 2000
	opt = ADAM(learn_rate)
	D = 8

	println("test sampler_a...")
	rbm = MPS(Float64, L, D=D)
	rbm_a = copy(rbm)

	x0, re = Flux.destructure(rbm_a)
	losses_a = Float64[]
	for i in 1:epoches
	    verbosity = (i % 100 == 0) ? 1 : 0
	    train_loss, grad = energy_and_grad(ham, rbm_a, sampler_a, n_chain=10, λ=1.0e-6, verbosity=verbosity)

	    Optimise.update!(opt, x0, grad)
	    rbm_a = re(x0)

	    push!(losses_a, train_loss)
	    if i % 100 == 0
	        println("energy at the $i-th step is $(train_loss).")
	    end
	end

	energy_a = energy(ham, rbm_a, sampler_a)
	energy_b = energy(ham, rbm_a, sampler_b)
	energy_c = energy(ham, rbm_a, sampler_c)
	println("energy_a=", energy_a, ", energy_b=", energy_b, ", energy_c=", energy_c)

	println("test sampler_b...")
	rbm_b = copy(rbm)

	x0, re = Flux.destructure(rbm_b)
	losses_b = Float64[]
	for i in 1:epoches
	    verbosity = (i % 100 == 0) ? 1 : 0
	    train_loss, grad = energy_and_grad(ham, rbm_b, sampler_b, n_chain=10, λ=1.0e-6, verbosity=verbosity)

	    Optimise.update!(opt, x0, grad)
	    rbm_b = re(x0)

	    push!(losses_b, train_loss)
	    if i % 100 == 0
	        println("energy at the $i-th step is $(train_loss).")
	    end
	end

	energy_a = energy(ham, rbm_b, sampler_a)
	energy_b = energy(ham, rbm_b, sampler_b)
	energy_c = energy(ham, rbm_b, sampler_c)
	println("energy_a=", energy_a, ", energy_b=", energy_b, ", energy_c=", energy_c)

	println("test sampler_c...")
	rbm_c = copy(rbm)

	x0, re = Flux.destructure(rbm_c)
	losses_c = Float64[]
	for i in 1:epoches
	    verbosity = (i % 100 == 0) ? 1 : 0
	    train_loss, grad = energy_and_grad(ham, rbm_c, sampler_c, n_chain=10, λ=1.0e-6, verbosity=verbosity)

	    Optimise.update!(opt, x0, grad)
	    rbm_c = re(x0)

	    push!(losses_c, train_loss)
	    if i % 100 == 0
	        println("energy at the $i-th step is $(train_loss).")
	    end
	end

	energy_a = energy(ham, rbm_c, sampler_a)
	energy_b = energy(ham, rbm_c, sampler_b)
	energy_c = energy(ham, rbm_c, sampler_c)
	println("energy_a=", energy_a, ", energy_b=", energy_b, ", energy_c=", energy_c)
	
end

