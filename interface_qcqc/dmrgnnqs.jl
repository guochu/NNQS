push!(LOAD_PATH, "/Users/guochu/Documents/QuantumSimulator/Fermions/src")
push!(LOAD_PATH, "../src")
include("./mol_hamiltonian.jl")


using Fermions
using NNQS
using Flux, Flux.Optimise


using JSON, Serialization

function read_data(pathname)
	# pathname = "lih.json"
	data = JSON.parsefile(pathname)
	E0 = data["E0"]
	L = data["L"]
	t = data["t"]
	t = [t...]
	v = data["v"]
	v = [v...]
	return E0, reshape(t, (L, L)), reshape(v, (L, L, L, L))
end

# use calc_ham_coeff.py to generate .json hamiltonian

function run_dmrg(D)
	pathname = "data/lih.json"
	E0, t, v = read_data(pathname)
	h1e, h2e = Fermions.get_spin_orbitals(t, 0.5*v)
	h2e′ = Fermions.remove_antisymmetric(Fermions.antisymmetrize(h2e))
	mpo = MPO(Fermions.qcmpo_spin_orbitals(h1e, h2e′))
	eigvalue, eigvector = ground_state(mpo, DMRG(D=D))
	energy = eigvalue + E0		
	println("DMRG energy=$(energy) for D=$(D)")

	mpspath = dirname(pathname) * "/mps_D$(D).txt"
	println("save mps to path ", mpspath)
	Serialization.serialize(mpspath, eigvector.data)
end

function run_nnqs(D)
	mpsdata = Serialization.deserialize("data/mps_D$(D).txt")
	mps0 = NNQS.MPS(mpsdata)
	nqs = NNQS.increase_bond!(mps0, D=D+5)

	# nnqs
    data_path = "mol_ham_data/lih/"

    op = read_binary_qubit_op( data_path * "qubit_op.data")
    ham = MolecularHamiltonian(op)
    op_n = read_binary_qubit_op( data_path * "qubit_op_n.data")
    ham_n = MolecularHamiltonian(op_n)
    op_s = read_binary_qubit_op( data_path * "qubit_op_s.data")
    ham_s = MolecularHamiltonian(op_s)
    # println(ham_s)
    L = ham.n_qubits
	n_hidden = L
	n_visible = L
    println("total number of qubits $L")
	# Random.seed!(3467891)

    n_sample = 10000
    n_chain_per_rank = 10
    n_sample_per_chain = div(n_sample, n_chain_per_rank)
	sampler = AutoRegressiveSampler(n_visible, n_sample_per_chain=n_sample_per_chain)	
    # sampler = Metropolis(n_visible, n_thermal=20000, n_sample_per_chain=4000, n_discard=100, mover=BondSwap(charge=0))
    # sampler = Metropolis(n_visible, n_thermal=10000, n_sample_per_chain=2000, n_discard=100, mover=BitFlip())


	# gradient descent algorithm using ADAM 

	learn_rate = 0.01
	epoches = 200
	opt = ADAM(learn_rate)

	x0, re = Flux.destructure(nqs)
    println("Number of parameters $(length(x0))")

	losses = Float64[]

    seeds = rand(100:10000, n_chain_per_rank)

    λt(n) = 1.0e-3 * 10.0^(-n / 100) + 1.0e-6
    for i in 1:epoches
        λ = λt(i)
        println("regularization at the $i-th step is $λ")
        # @time train_loss, grad = energy_and_grad_sr(ham, sampler, nqs, n_chain_per_rank=n_chain_per_rank, seeds=seeds, diag_shift=0.01, λ=λ)
        @time train_loss, grad = energy_and_grad(ham, sampler, nqs, n_chain_per_rank=n_chain_per_rank, seeds=seeds, λ=1.0e-5)

        Optimise.update!(opt, x0, grad)
        nqs = re(x0)

        push!(losses, real(train_loss))
        println("energy at the $i-th step is $(train_loss).")

        if i % 50 == 0
            n = energy(ham_n, sampler, nqs)
            s = energy(ham_s, sampler, nqs)
            println("    particle_num (2.0 for H2): $(n)")
            println("    total_spin (0.0 for H2): $(s)")
        end
    end
    return losses
end

