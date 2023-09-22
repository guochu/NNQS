

struct BatchAutoRegressiveSampler{T<:AbstractConstrain} <: AbstractSampler
	constrain::T
	N::Int
	n_sample_per_chain::Int
end

BatchAutoRegressiveSampler(constrain::AbstractConstrain, N::Int; n_sample_per_chain::Int=500) = BatchAutoRegressiveSampler(constrain, N, n_sample_per_chain)
BatchAutoRegressiveSampler(N::Int; constrain::AbstractConstrain=NoConstrain(), kwargs...) = BatchAutoRegressiveSampler(constrain, N; kwargs...)

function batchautoregressivesampling(nnqs::MPS, n::Int, constrain::NoConstrain)
	@assert length(nnqs) > 0
	@assert isrightcanonical(nnqs)
	psi = nnqs.data
	m1 = psi[1][:,1,:]
	p1 = real(dot(m1, m1)) # probability of state 1 on site 1
	# println(p1, " ", real(dot(psi[1][:,2,:], psi[1][:,2,:])))
	# perform local sampling 
	physical_states = [1, -1]
	physical_indices = [1, 2]
	prob = [p1, 1-p1]
	dist = Multinomial(n, prob)
	# the following triple keeps all the information
	unique_samples = [[1], [-1]]
	counts = rand(dist)
	leftstorages = [LinearAlgebra.rmul!(psi[1][:,i,:], 1/sqrt(prob[i])) for i in 1:2]

	# probability of s1
	for i in 2:length(psi)
		# println("i=", i)
		counts_tmp = Int[]
		unique_samples_tmp = Vector{Int}[]
		leftstorages_tmp = Matrix{eltype(nnqs)}[]
		for (unique_sample, count, m) in zip(unique_samples, counts, leftstorages)
			m1 = m * psi[i][:, 1, :]
			p1 = real(dot(m1, m1))
			prob = [p1, 1-p1]
			dist = Multinomial(count, prob)
			subtree_count = rand(dist)
			for (j, count_j, s_j, p_j) in zip(physical_indices, subtree_count, physical_states, prob)
				if count_j != 0
					push!(counts_tmp, count_j)
					push!(unique_samples_tmp, vcat(unique_sample, s_j))
					push!(leftstorages_tmp, LinearAlgebra.mul!(similar(m1), m, psi[i][:,j,:], 1/sqrt(p_j), false))
				end
			end
		end
		counts = counts_tmp
		unique_samples = unique_samples_tmp
		leftstorages = leftstorages_tmp
	end
	return vv2m(unique_samples), counts
end


function batchautoregressivesampling(nnqs::MPS, n::Int, constrain::U1Constrain)
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

function batchautoregressivesampling(nnqs::MPS, n::Int, constrain::U1U1Constrain)
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

function vv2m(v::Vector{<:Vector})
	@assert !isempty(v)
	L = length(v)
	N = length(v[1])
	r = zeros(Int, N, L)
	for i in 1:L
		r[:, i] = v[i]
	end
	return r
end
