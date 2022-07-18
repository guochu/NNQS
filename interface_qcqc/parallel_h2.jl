
@everywhere include("./mol_hamiltonian.jl")
include("../parallel/parallel_nqs.jl")


using Random
using Flux, Flux.Optimise




function test_gs_energy()

    molecule = "lih"

    data_path = "mol_ham_data/$molecule/"

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
	# rbm = FCN(Float64, n_hidden=n_hidden, n_visible=n_visible, activation=tanh)
    rbm = FCN(Float64, n_hidden=n_hidden, n_visible=n_visible)
    # real rbm
    # rbm = FCN(ComplexF64, n_hidden=n_hidden, n_visible=n_visible, activation=exp)

    n_sample = 20000
    n_chain = nworkers()
    n_sample_per_chain = div(n_sample, n_chain)
    println("n_sample_per_chain = $n_sample_per_chain")

	sampler = Metropolis(n_visible, n_thermal=20000, n_sample_per_chain=n_sample_per_chain, n_discard= 100, mover=FermiBondSwap((-2,-2)))	
    # sampler = Metropolis(n_visible, n_thermal=20000, n_sample_per_chain=4000, n_discard=100, mover=BondSwap(charge=0))
    # sampler = Metropolis(n_visible, n_thermal=10000, n_sample_per_chain=2000, n_discard=100, mover=BitFlip())


	# gradient descent algorithm using ADAM 

	learn_rate = 0.01
	epoches = 200
	opt = ADAM(learn_rate)

    paras = Flux.params(rbm)
	x0 = parameters(paras)
    
    println("Number of parameters $(length(x0))")

	losses = Float64[]

    λt(n) = 1.0e-3 * 10.0^(-n / 100) + 1.0e-6
    for i in 1:epoches
        λ = λt(i)
        println("regularization at the $i-th step is $λ")
        @time train_loss, grad = parallel_energy_and_grad_sr(ham, sampler, rbm, n_chain=n_chain, diag_shift=0.01, λ=λ)
        # @time train_loss, grad = parallel_energy_and_grad(ham, sampler, rbm, n_chain=n_chain, λ=1.0e-5)


        Optimise.update!(opt, x0, grad)
        reset!(paras, x0)

        push!(losses, real(train_loss))
        println("energy at the $i-th step is $(train_loss).")

        if i % 50 == 0
            n = energy(ham_n, sampler, rbm)
            s = energy(ham_s, sampler, rbm)
            println("    particle_num (2.0 for $molecule): $(n)")
            println("    total_spin (0.0 for $molecule): $(s)")
        end
    end
    return losses
end

test_gs_energy()
