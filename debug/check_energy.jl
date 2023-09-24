include("../src/includes.jl")

# push!(LOAD_PATH, "../src")
# using NNQS

using QuantumSpins

function ising_chain2(L::Int; J::Real, hz::Real)
	p = spin_half_matrices()
    x, z = p["x"], p["z"]	
    terms = []
    for i in 1:L
    	push!(terms, QTerm(i=>x, coeff=hz))
    end
    for i in 1:L-1
    	push!(terms, QTerm(i=>z, i+1=>z, coeff=J))
    end
    return QuantumOperator([terms...])
end

function check_energy()
	L = 10
	J = 1.
	hz = 0.7
	h = ising_chain2(L, J=J, hz = hz)
	mpo = MPO(h)
	eigvalue, eigvector = ground_state(mpo, DMRG(D=4))
	println("DMRG energy is ", eigvalue)

	nqs = increase_bond!(MPS(eigvector.data), D=8)
	h = IsingChain(h=hz, J=J)
	println("nnqs exact energy is ", energy_exact(h, nqs))

	n_sample_per_chain = 5000
	sampler = AutoRegressiveSampler(L, n_sample_per_chain=n_sample_per_chain)	
	println("nnqs autoregressive energy is ", energy(h, nqs, sampler))

	sampler = BatchAutoRegressiveSampler(L, n_sample_per_chain=n_sample_per_chain)	
	println("nnqs batchautoregressive energy is ", energy(h, nqs, sampler))

	sampler = MetropolisLocal(L, n_thermal=10000, n_sample_per_chain=n_sample_per_chain, n_discard=100)	
	println("nnqs MCMC energy is ", energy(h, nqs, sampler))

end


function check_grad()
	L = 3
	J = 1.
	hz = 0.7
	nqs = MPS(ComplexF64, L, D=2)
	h = IsingChain(h=hz, J=J)
	_energy, grad = energy_and_grad_exact(h, nqs)
	println("nnqs exact energy and grad is ", _energy, " ", norm(grad))

	n_sample_per_chain = 10000000000000

	# sampler = AutoRegressiveSampler(L, n_sample_per_chain=n_sample_per_chain)	
	# _energy, back = Zygote.pullback(energy, h, nqs, sampler)
	# _a, _grad, _b = back(one(_energy))
	# grad1, re = Flux.destructure(_grad)
	# println("nnqs autoregressive energy is ", _energy, " ", norm(grad-grad1) / norm(grad))

	sampler = BatchAutoRegressiveSampler(L, n_sample_per_chain=n_sample_per_chain)	
	# _energy, back = Zygote.pullback(energy, h, nqs, sampler)
	# _a, _grad, _b = back(one(_energy))
	# grad2, re = Flux.destructure(_grad)
	_energy, grad2 = energy_and_grad(h, nqs, sampler)
	println("nnqs batchautoregressive energy is ", _energy, " ",  norm(grad-grad2) / norm(grad) )

	# sampler = MetropolisLocal(L, n_thermal=10000, n_sample_per_chain=n_sample_per_chain, n_discard=100)	
	# _energy, back = Zygote.pullback(energy, h, nqs, sampler)
	# _a, _grad, _b = back(one(_energy))
	# grad3, re = Flux.destructure(_grad)
	# println("nnqs MCMC energy is ", _energy, " ",  norm(grad-grad3) / norm(grad) )

end


