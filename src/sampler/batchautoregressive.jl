

struct BatchAutoRegressiveSampler{T<:AbstractConstrain} <: AbstractSampler
	constrain::T
	N::Int
	n_sample_per_chain::Int
end

BatchAutoRegressiveSampler(constrain::AbstractConstrain, N::Int; n_sample_per_chain::Int=500) = BatchAutoRegressiveSampler(constrain, N, n_sample_per_chain)
BatchAutoRegressiveSampler(N::Int; constrain::AbstractConstrain=NoConservation(), kwargs...) = BatchAutoRegressiveSampler(constrain, N; kwargs...)

function batchautoregressivesampling(nnqs::MPS, n::Int, constrain::NoConservation)
	@assert length(nnqs) > 0
	@assert isrightcanonical(nnqs)
	psi = nnqs.data
	m1 = psi[1][:,1,:]
	p1 = _normalize(real(dot(m1, m1))) # probability of state 1 on site 1
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
			p1 = _normalize(real(dot(m1, m1)))
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


function batchautoregressivesampling(nnqs::MPS, n::Int, constrain::LeafConservationConstrain)
	@assert length(nnqs) > 0
	@assert isrightcanonical(nnqs)
	psi = nnqs.data
	m1 = psi[1][:,1,:]
	p1 = _normalize(real(dot(m1, m1))) # probability of state 1 on site 1
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
			p1 = _normalize(real(dot(m1, m1)))
			prob = [p1, 1-p1]
			dist = Multinomial(count, prob)
			subtree_count = rand(dist)
			for (j, count_j, s_j, p_j) in zip(physical_indices, subtree_count, physical_states, prob)
				if count_j != 0
					unique_sample_new = vcat(unique_sample, s_j)
					if satisfied(constrain, unique_sample_new)
						push!(counts_tmp, count_j)
						push!(unique_samples_tmp, unique_sample_new)
						push!(leftstorages_tmp, LinearAlgebra.mul!(similar(m1), m, psi[i][:,j,:], 1/sqrt(p_j), false))
					end
				end
			end
		end
		counts = counts_tmp
		unique_samples = unique_samples_tmp
		leftstorages = leftstorages_tmp
	end
	return vv2m(unique_samples), counts
end

function batchautoregressivesampling(nnqs::MPS, n::Int, constrain::NodeConservationConstrain)
	@assert length(nnqs) > 0
	@assert isrightcanonical(nnqs)
	psi = nnqs.data
	m1 = psi[1][:,1,:]
	p1 = _normalize(real(dot(m1, m1))) # probability of state 1 on site 1
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
			p1 = _normalize(real(dot(m1, m1)))
			prob = [p1, 1-p1]
			all_unique_sample_new = [vcat(unique_sample, s_j) for s_j in physical_states]
			prob_new = [satisfied(constrain, sample) ? p : zero(p) for (p, sample) in zip(prob, all_unique_sample_new)]
			prob_new ./= sum(prob_new)
			dist = Multinomial(count, prob_new)
			subtree_count = rand(dist)
			for (j, count_j, s_j, p_j) in zip(physical_indices, subtree_count, physical_states, prob)
				if count_j != 0
					unique_sample_new = vcat(unique_sample, s_j)
					push!(counts_tmp, count_j)
					push!(unique_samples_tmp, unique_sample_new)
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

function _normalize(p::Real, tol::Real=1.0e-12)
	if abs(p) < tol
		return zero(p)
	elseif abs(p-1) < tol
		return one(p)
	else
		return p
	end
end