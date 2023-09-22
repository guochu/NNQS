# converting -1 and 1 into 2 and 1
index_to_state(index) = [(index[i] == 2) ? -1 : 1 for i in 1:length(index)]
state_to_index(state) = CartesianIndex(Tuple((item == -1) ? 2 : 1 for item in state))

# function Ψ_exact(nnqs::AbstractNNQS)
# 	L = sys_size(nnqs)
# 	v = zeros(eltype(nnqs), 2^L)
# 	p_all = 0.
# 	indices = CartesianIndices(ntuple(x->2, L))
# 	c = 1
# 	for index in indices
# 		state = index_to_state(index)
# 	end
# end

function matrix(h::Hamiltonian, L::Int)
	indices = CartesianIndices(ntuple(x->2, L))
	linear_indices = LinearIndices(ntuple(x->2, L))
	m = zeros(eltype(h), length(indices), length(indices))
	for index in indices
		state = index_to_state(index)
		E_diag = diagonal_coupling(h, state)
		pos_i = linear_indices[index]
		m[pos_i, pos_i] = E_diag
		c_states, coefs = coupled_states(h, state)
		if !isempty(coefs)
			for n in 1:size(c_states, 2)
				index_j = state_to_index(c_states[:, n])
				pos_j = linear_indices[index_j]
				m[pos_j, pos_i] = coefs[n]
			end
		end
	end
	return m
end

function energy_exact(h::Hamiltonian, nnqs::AbstractNNQS)
	L = sys_size(nnqs)
	E_loc = zero(local_energy_single(h, nnqs, ones(Int, L)))
	indices = CartesianIndices(ntuple(x->2, L))
	p_all = 0.
	for index in indices
		state = index_to_state(index)
		E_loc_single = local_energy_single(h, nnqs, state)
		amp = Ψ(nnqs, state)
		p_single = abs2(amp)
		p_all += p_single
		E_loc += E_loc_single * p_single
	end
	return E_loc / p_all
end

function energy_and_grad_exact(h::Hamiltonian, nnqs::AbstractNNQS; λ::Real = 1.0e-6)
	L = sys_size(nnqs)
	E_loc, ∂θ = local_energy_and_grad(h, nnqs, ones(Int, L))

	E_loc = zero(E_loc)
	∂θ = zero(∂θ)
	E∂θ = zero(∂θ)

	indices = CartesianIndices(ntuple(x->2, L))
	p_all = 0.
	for index in indices
		state = index_to_state(index)
		E_loc_batch, ∂θ_batch = local_energy_and_grad(h, nnqs, state)
		amp = Ψ(nnqs, state)
		p_single = abs2(amp)
		# println("probability for basis $index is $p_single")
		E_loc += E_loc_batch * p_single
		∂θ .+= ∂θ_batch * p_single
		E∂θ .+= (E_loc_batch * p_single) .* ∂θ_batch 
		# ∂θ∂θ .+= reshape(kron(conj(∂θ_batch), ∂θ_batch), _nparas, _nparas)
		p_all += p_single
	end
	E_loc /= p_all
	∂θ ./= p_all
	E∂θ ./= p_all

	return E_loc, _regularization!(2 .* (E∂θ .- real(E_loc) .* ∂θ), Flux.destructure(nnqs)[1], λ)
end

function local_energy_and_grad(h::Hamiltonian, nnqs::AbstractNNQS, state::ComputationBasis)
	# amp_state, back = Zygote.pullback( () -> Ψ(nnqs, state), Flux.params(nnqs) )
	amp_state, back = Zygote.pullback( m -> Ψ(m, state), nnqs )
	_energy = local_energy_single(h, nnqs, state, amp_state)
	grad, re = Flux.destructure(back(one(_energy)))
	grad ./= conj(amp_state)
	return _energy, grad
end

# function energy_and_grad_sr_exact(h::Hamiltonian, nnqs::AbstractNNQS; diag_shift::Real=1.0e-4, λ::Real = 1.0e-6)
# 	L = sys_size(nnqs)
# 	E_loc, ∂θ = local_energy_and_grad(h, nnqs, ones(Int, L))

# 	E_loc = zero(E_loc)
# 	∂θ = zero(∂θ)
# 	E∂θ = zero(∂θ)
# 	∂θ∂θ = zeros(eltype(∂θ), length(∂θ), length(∂θ))

# 	indices = CartesianIndices(ntuple(x->2, L))
# 	p_all = 0.
# 	for index in indices
# 		state = index_to_state(index)
# 		E_loc_batch, ∂θ_batch = local_energy_and_grad(h, nnqs, state)
# 		amp = Ψ(nnqs, state)
# 		p_single = abs2(amp)
# 		E_loc += E_loc_batch * p_single
# 		∂θ .+= ∂θ_batch * p_single
# 		E∂θ .+= (E_loc_batch * p_single) .* ∂θ_batch
# 		∂θ∂θ .+= reshape(kron(conj(∂θ_batch), ∂θ_batch), length(∂θ), length(∂θ)) * p_single
# 		# push!(energies, E_loc_batch)
# 		p_all += p_single
# 	end
# 	E_loc /= p_all
# 	∂θ ./= p_all
# 	E∂θ ./= p_all
# 	∂θ∂θ ./= p_all
# 	grad = 2 .* (E∂θ .- real(E_loc) .* ∂θ)
# 	S = ∂θ∂θ - reshape(kron(conj(∂θ), ∂θ), length(∂θ), length(∂θ))
# 	# println("average sign is $(average_sign(energies))")

#  	# # regularization
#  	# S .+= reg .* Diagonal(S)
#  	# # grad_sr = S \ grad
#  	# grad_sr = pinv(S) * grad
#  	# return E_loc, grad_sr

#  	# return E_loc, stable_solve(S, grad, diag_shift=diag_shift)
# 	return E_loc, _regularization!(stable_solve(S, grad, diag_shift=diag_shift), Flux.destructure(nnqs)[1], λ)
# end
