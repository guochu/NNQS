include("./mol_hamiltonian.jl")


using Random
using NNQS
using Flux, Flux.Optimise




function test_gs_energy()

    data_path = "mol_ham_data/lih/"

    op = read_binary_qubit_op( data_path * "qubit_op.data")
    ham = MolecularHamiltonian(op)
    op_n = read_binary_qubit_op( data_path * "qubit_op_n.data")
    ham_n = MolecularHamiltonian(op_n)
    op_s = read_binary_qubit_op( data_path * "qubit_op_s.data")
    ham_s = MolecularHamiltonian(op_s)
    # println(ham_s)
    L = ham.n_qubits
	n_hidden = 2*L
	n_visible = L
    println("total number of qubits $L")
	# Random.seed!(3467891)

    n_sample = 10000
    n_chain_per_rank = 10
    n_sample_per_chain = div(n_sample, n_chain_per_rank)
	rbm = FCN(Float64, n_hidden=n_hidden, n_visible=n_visible, activation=tanh)
	sampler = Metropolis(n_visible, n_thermal=20000, n_sample_per_chain=n_sample_per_chain, n_discard= 100, mover=FermiBondSwap((-2,-2)))	
    # sampler = Metropolis(n_visible, n_thermal=20000, n_sample_per_chain=4000, n_discard=100, mover=BondSwap(charge=0))
    # sampler = Metropolis(n_visible, n_thermal=10000, n_sample_per_chain=2000, n_discard=100, mover=BitFlip())


	# gradient descent algorithm using ADAM 

	learn_rate = 0.01
	epoches = 500
	opt = ADAM(learn_rate)

    paras = Flux.params(rbm)
	x0 = parameters(paras)
    println("Number of parameters $(length(x0))")

	losses = Float64[]

    seeds = rand(100:10000, n_chain_per_rank)

    λt(n) = 1.0e-3 * 10.0^(-n / 100) + 1.0e-6
    for i in 1:epoches
        λ = λt(i)
        println("regularization at the $i-th step is $λ")
        @time train_loss, grad = energy_and_grad_sr(ham, sampler, rbm, n_chain_per_rank=n_chain_per_rank, seeds=seeds, diag_shift=0.01, λ=λ)
        # @time train_loss, grad = energy_and_grad(ham, sampler, rbm, n_chain_per_rank=n_chain_per_rank, seeds=seeds, λ=1.0e-5)

        Optimise.update!(opt, x0, grad)
        reset!(paras, x0)

        push!(losses, real(train_loss))
        println("energy at the $i-th step is $(train_loss).")

        if i % 50 == 0
            n = energy(ham_n, sampler, rbm)
            s = energy(ham_s, sampler, rbm)
            println("    particle_num (2.0 for H2): $(n)")
            println("    total_spin (0.0 for H2): $(s)")
        end
    end
    return losses
end

# test_gs_energy()