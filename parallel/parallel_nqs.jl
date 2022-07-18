using Distributed, Flux, Flux.Optimise, Distributions

@everywhere push!(LOAD_PATH, "../src")

@everywhere using NNQS, Random

# # the input is a vector of energies and gradients
# function _regroup_energies_and_grads(r::Vector{<:EnergiesGrads})
# 	_energies = copy(r[1].energies)
# 	_grads = r[1].grads
# 	for i in 2:length(r)
# 		append!(_energies, r[i].energies)
# 		append!(_grads, r[i].grads)
# 	end
# 	return EnergiesGrads(_energies, _grads), [item.energies for item in r]
# end

# the parallel part
function parallel_energy_and_grad_util(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; n_chain::Int=10, verbosity::Int=1)
	n_rank = nworkers()
	n_chain_per_rank = div(n_chain, n_rank)
	(verbosity >= 2) && println("n_chain_per_rank = $n_chain_per_rank")
	if n_chain % n_rank != 0
		new_n_chain = n_chain_per_rank * n_rank
		println("n_chain=$n_chain can not be divided by n_rank=$n_rank, change n_chain to $new_n_chain")
		n_chain = new_n_chain
	end
	all_seeds = [rand(1000:100000, n_chain_per_rank) for i in 1:n_rank]
	f_per_rank(i::Int) = NNQS.energy_and_grad_util_per_rank(h, sampler, nnqs, n_chain_per_rank=n_chain_per_rank, seeds=all_seeds[i])

	r = pmap(f_per_rank, 1:n_rank)
	ra = r[1]
	for i in 2:length(r)
		append!(ra, r[i])
	end
	return 	ra
end

function parallel_energy_and_grad(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; n_chain::Int=10, λ::Real = 1.0e-6, verbosity::Int=1)

	energies_∂θs = parallel_energy_and_grad_util(h, sampler, nnqs; n_chain=n_chain, verbosity=verbosity)

	# the following code are not parallelized

	println("total number of samples is $(sum([length(energies_∂θs[i].energies) for i in 1:length(energies_∂θs)]))")

	E_loc, grad = NNQS.compute_energy_and_grad(aggregate(energies_∂θs), parameters(Flux.params(nnqs)), λ=λ)

	if verbosity >= 1
		mean_energies = [mean(energies_∂θs[i].energies) for i in 1:length(energies_∂θs)]
		println("Ē = $E_loc ± $(std(mean_energies))")
	end

	return E_loc, grad

end

function parallel_energy_and_grad_sr(h::Hamiltonian, sampler::AbstractSampler, nnqs::AbstractNNQS; n_chain::Int=nworkers(), diag_shift::Real=1.0e-2, λ::Real = 1.0e-6, verbosity::Int=1)

	energies_∂θs = parallel_energy_and_grad_util(h, sampler, nnqs; n_chain=n_chain, verbosity=verbosity)

	# the following code are not parallelized

	println("total number of samples is $(sum([length(energies_∂θs[i].energies) for i in 1:length(energies_∂θs)]))")

	E_loc, grad = NNQS.compute_energy_and_grad_sr(aggregate(energies_∂θs), parameters(Flux.params(nnqs)), diag_shift=diag_shift, λ=λ)

	if verbosity >= 1
		mean_energies = [mean(energies_∂θs[i].energies) for i in 1:length(energies_∂θs)]
		println("Ē = $E_loc ± $(std(mean_energies))")
	end

	return E_loc, grad

end




function simple_test(L; h, J)
	ham = NNQS.IsingChain(h=h, J=J)
	n_hidden = 2 * L
	n_visible = L
	rbm = FCN(Float64, n_hidden=n_hidden, n_visible=n_visible, activation=tanh)
	sampler = MetropolisLocal(n_visible, n_thermal=10000, n_sample_per_chain=5, n_discard= 100)

	learn_rate = 0.01
	epoches = 1000

	opt = ADAM(learn_rate)

	paras = Flux.params(rbm)
	x0 = parameters(paras)

	losses = Float64[]


    for i in 1:epoches
        train_loss, grad = parallel_energy_and_grad(ham, sampler, rbm, λ=1.0e-5, n_chain=nworkers())
        # train_loss, grad = energy_and_grad(ham, sampler, rbm, λ=0.)

        Optimise.update!(opt, x0, grad)
        reset!(paras, x0)

        push!(losses, real(train_loss))

        println("energy at the $i-th step is $(train_loss)")
    end

    return losses, losses_exact
end

