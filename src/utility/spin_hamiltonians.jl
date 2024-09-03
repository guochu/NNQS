struct IsingChain <: Hamiltonian
	h::Float64
	J::Float64
end

IsingChain(; h::Real, J::Real) = IsingChain(convert(Float64, h), convert(Float64, J))
Base.eltype(::Type{IsingChain}) = Float64

function coupled_states(h::IsingChain, state::ComputationBasis)
	L = length(state)
	c_states = zeros(Int, L, L)
	coefs = zeros(Float64, L)
	# c_states = Tuple{ComputationBasis, Float64}[]
	for j in 1:L
		new_state = copy(state)
		new_state[j] = -state[j]
		c_states[:, j] = new_state
		coefs[j] = h.h
		# push!(c_states, new_state)
		# push!(coefs, h.h)
	end
	return c_states, coefs
end

function diagonal_coupling(h::IsingChain, state::ComputationBasis)
	L = length(state)
	E_diag = 0.
	for j in 1:L-1
		E_diag += h.J * state[j] * state[j+1]
	end
	return E_diag
end

struct Ising2D <: Hamiltonian
	shape::Tuple{Int, Int}
	h::Float64
	J::Float64
	periodic::Bool
end
Ising2D(shape::Tuple{Int, Int}; h::Real, J::Real, periodic::Bool=false) = Ising2D(shape, convert(Float64, h), convert(Float64, J), periodic)
Base.eltype(::Type{Ising2D}) = Float64

function coupled_states(h::Ising2D, state::ComputationBasis)
	L = length(state)
	@assert prod(h.shape) == L
	c_states = zeros(Int, L, L)
	coefs = zeros(Float64, L)
	for j in 1:L
		new_state = copy(state)
		new_state[j] = -state[j]
		c_states[:, j] = new_state
		coefs[j] = h.h
	end
	return c_states, coefs
end

function diagonal_coupling(h::Ising2D, state::ComputationBasis)
	@assert prod(h.shape) == length(state)
	m, n = h.shape
	index = LinearIndices(h.shape)
	E_diag = 0.
	if h.periodic
		for i in 1:m, j in 1:n
			idx1, idx2 = index[i, j], index[mod1(i+1, m), j]
			E_diag += h.J * state[idx1] * state[idx2]
			idx2 = index[i, mod1(j+1, n)]
			E_diag += h.J * state[idx1] * state[idx2]
		end
	else
		for i in 1:m
			for j in 1:n-1
				idx1, idx2 = index[i, j], index[i, j+1]
				E_diag += h.J * state[idx1] * state[idx2]
			end
		end
		for j in 1:n
			for i in 1:m-1
				idx1, idx2 = index[i, j], index[i+1, j]
				E_diag += h.J * state[idx1] * state[idx2]
			end
		end
	end
	return E_diag
end

struct HeisenbergChain <: Hamiltonian
	J::Float64
	Jzz::Float64
	h::Float64
end

HeisenbergChain(; J::Real, Jzz::Real, h::Real) = HeisenbergChain(convert(Float64, J), convert(Float64, Jzz), convert(Float64, h))
Base.eltype(::Type{HeisenbergChain}) = Float64

function coupled_states(h::HeisenbergChain, state::ComputationBasis)
	L = length(state)
	c_states = Vector{Int}[]
	coefs = Float64[]
	for j in 1:L-1
		if (state[j]==-1) && (state[j+1] == 1)
			new_state = copy(state)
			new_state[j] = 1
			new_state[j+1] = -1
			push!(c_states, new_state)
			push!(coefs, 2*h.J)
		end
		if (state[j]==1) && (state[j+1] == -1)
			new_state = copy(state)
			new_state[j] = -1
			new_state[j+1] = 1
			push!(c_states, new_state)
			push!(coefs, 2*h.J)
		end		
	end
	return hcat(c_states...), coefs
end

function diagonal_coupling(h::HeisenbergChain, state::ComputationBasis)
	L = length(state)
	E_diag = 0.
	for j in 1:L-1
		E_diag += h.Jzz * state[j] * state[j+1]
	end
	for j in 1:L
		E_diag += h.h * state[j]
	end
	return E_diag
end


