# autoregressive sampling

struct AutoRegressiveSampler <: AbstractSampler
	N::Int
	n_sample_per_chain::Int
end
AutoRegressiveSampler(N::Int; n_sample_per_chain::Int=500) = AutoRegressiveSampler(N, n_sample_per_chain)

autoregressivesampling(nnqs::AbstractNNQS) = error("autoregressivesampling not implemented for nqs type $(typeof(nnqs))")

function autoregressivesampling(nnqs::MPS)
	@assert length(nnqs) > 0
	@assert isrightcanonical(nnqs)
	psi = nnqs.data
	m1 = psi[1][:,1,:]
	p1 = real(dot(m1, m1)) # probability of state 1 on site 1
	# println(p1, " ", real(dot(psi[1][:,2,:], psi[1][:,2,:])))
	# perform local sampling 
	s1 = (rand() < p1) ? 1 : 2
	state = [(s1==1) ? 1 : -1]
	# probability of s1
	prob_s1 = (s1 == 1) ? p1 : 1-p1
	mm = LinearAlgebra.rmul!(psi[1][:,s1,:], 1/sqrt(prob_s1))
	for i in 2:length(psi)
		m1 = mm * psi[i][:, 1, :]
		p1 = real(dot(m1, m1))
		# println(p1, " ", real(dot(mm * psi[i][:, 2, :], mm * psi[i][:, 2, :])))
		s1 = (rand() < p1) ? 1 : 2
		prob_s1 = (s1 == 1) ? p1 : 1-p1
		mm = LinearAlgebra.rmul!(mm * psi[i][:,s1,:], 1/sqrt(prob_s1))
		push!(state, (s1==1) ? 1 : -1 )
	end
	return state
end
