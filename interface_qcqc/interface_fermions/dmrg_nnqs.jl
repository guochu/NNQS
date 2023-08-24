### Change the following paths according to the environment.
using Pkg
Pkg.activate("/home/aurora/Tests/Quantum/NNQS/NNQS-env")
using Fermions
include("/home/aurora/Softwares/Fermions/models/LiH/util.jl")
using NNQS
include("/home/aurora/Softwares/NNQS/interface_qcqc/mol_hamiltonian.jl")
using Random
using Flux, Flux.Optimise

function select_terms(terms::Vector, k::Int)
    r = []
    for t in terms
        if k in positions(t)
            push!(r, t)
        end
    end
    return r
end

function get_ham(data_path::String)
    E0, t, v = read_data(data_path)
    lattice = SpinlessFermion()
    h = hamiltonian(lattice, t, v)
    return simplify(h)
end

function do_dmrg()
    E0, t, v = read_data("ham_coeff_for_fermions.json")
    println("E0 is $E0")
    println("number of orbitals is $(size(t, 1))")

    h1e, h2e = Fermions.get_spin_orbitals(t, 0.5*v)
    h2e′ = Fermions.remove_antisymmetric(Fermions.antisymmetrize(h2e))
    @time mpo = MPO(Fermions.qcmpo_spin_orbitals(h1e, h2e′))

    alg = SVDCompression(D=1000, tol=1.0e-10)
    println("mpo bond dimension ", bond_dimensions(mpo))
    compress!(mpo, alg=alg)

    println("mpo bond dimension ", bond_dimensions(mpo))

    # DMRG calculations
    D = 10
    maxiter = 10
    alg = DMRG(D=D, maxiter=maxiter, verbosity=3)
    eigvalue, eigvector = ground_state(mpo, alg)
    energy = eigvalue + E0    
    println("DMRG energy is $energy")

    return eigvalue, eigvector, E0
end

function do_nnqs()
    qubit_op = read_binary_qubit_op("ham_qubit_op_for_nnqs.data")
    ham = MolecularHamiltonian(qubit_op)
    L = ham.n_qubits
    n_hidden = L
    n_visible = L
    n_sample = 5000
    n_chain_per_rank = 10
    n_sample_per_chain = div(n_sample, n_chain_per_rank)
    rbm = FCN(Float64, n_hidden=n_hidden, n_visible=n_visible, activation=tanh)
    # sampler = AutoRegressiveSampler(L, n_sample_per_chain=n_sample_per_chain)
    sampler = Metropolis(n_visible, n_thermal=20000, n_sample_per_chain=n_sample_per_chain, n_discard= 100, mover=FermiBondSwap((0,0)))
    learn_rate = 0.01
    epoches = 200
    opt = ADAM(learn_rate)
    x0, re = Flux.destructure(rbm)
    println("Number of parameters $(length(x0))")
    losses = Float64[]
    seeds = rand(100:10000, n_chain_per_rank)
    λt(n) = 1.0e-3 * 10.0^(-n / 100) + 1.0e-6
    for i in 1:epoches
        λ = λt(i)
        println("regularization at the $i-th step is $λ")
        train_loss, grad = energy_and_grad_sr(ham, sampler, rbm, n_chain_per_rank=n_chain_per_rank, seeds=seeds, diag_shift=0.01, λ=λ)
        Optimise.update!(opt, x0, grad)
        rbm = re(x0)
        push!(losses, real(train_loss))
        println("energy at the $i-th step is $(train_loss).")
    end
end

function check_dmrg_and_nnqsmps()
    eigvalue, eigvector, E0 = do_dmrg()

    # Reverse the order of all orbitals (sites).
    mps_data_for_nnqs = copy(eigvector.data)
    for i in 1:length(mps_data_for_nnqs)
        tmp = copy(mps_data_for_nnqs[i][:, 1, :])
        mps_data_for_nnqs[i][:, 1, :] = mps_data_for_nnqs[i][:, 2, :]
        mps_data_for_nnqs[i][:, 2, :] = tmp
    end

    nqs = NNQS.MPS(mps_data_for_nnqs)

    qubit_op = read_binary_qubit_op("ham_qubit_op_for_nnqs.data")
    ham = MolecularHamiltonian(qubit_op)
    L = ham.n_qubits
    sampler = AutoRegressiveSampler(L, n_sample_per_chain=1024)
    println("NNQS autoregressive energy is ", NNQS.energy(ham, sampler, nqs))
    println("*Note: the above energy should be very close to the \
DMRG calculated energy which is $(eigvalue + E0).")

    # data_tmp = [zeros(1, 2, 1) for i in 1:L]
    # for i in 1:L
    #     data_tmp[i][1, 1, 1] = 1.
    #     data_tmp[i][1, 2, 1] = 0.
    # end
    # nqs_tmp = NNQS.MPS(data_tmp)
    # println("NNQS autoregressive energy is ", NNQS.energy(ham, sampler, nqs_tmp))
end

# do_nnqs()
check_dmrg_and_nnqsmps()
