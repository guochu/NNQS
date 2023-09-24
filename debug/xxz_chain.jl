include("../src/includes.jl")
# push!(LOAD_PATH, "../src")
# using NNQS


using Random, LinearAlgebra
using Flux, Flux.Optimise

function run_xxz_exact(L; h, J, Jzz)
	ham = HeisenbergChain(h=h, J=J, Jzz=Jzz)
	m = matrix(ham, L)
	evals = eigvals(Symmetric(m))
	println("ground state energy is ", evals[1])
end

function test1(L; h, J, Jzz)
	ham = HeisenbergChain(h=h, J=J, Jzz=Jzz)
	n_hidden = 2 * L
	n_visible = L
	Random.seed!(1234)
	rbm = MPS(Float64, n_visible, D=20)
	rightorth!(rbm)
	sampler = BatchAutoRegressiveSampler(n_visible, n_sample_per_chain=10000)

	learn_rate = 0.001
	epoches = 1000
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


function run_xxz(L; h, J, Jzz)
	ham = HeisenbergChain(h=h, J=J, Jzz=Jzz)
	n_hidden = 2 * L
	n_visible = L
	Random.seed!(1234)
	rbm = MPS(Float64, n_visible, D=20)
	rightorth!(rbm)
	sampler = BatchAutoRegressiveSampler(n_visible, n_sample_per_chain=10000, constrain=U1LeafConservation(L, c=div(L,2)))	
	# sampler = BatchAutoRegressiveSampler(n_visible, n_sample_per_chain=10000000000, constrain=U1NodeConservation(L, c=div(L,2)))
	sampler2 = BatchAutoRegressiveSampler(n_visible, n_sample_per_chain=10000)
	println("compare initial energy ", energy(ham, rbm, sampler), " ", energy(ham, rbm, sampler2))	
	# gradient descent algorithm using ADAM 

	learn_rate = 0.001
	epoches = 1000
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

    println("compare final energy ", energy(ham, rbm, sampler), " ", energy(ham, rbm, sampler2))	

    return losses
end
