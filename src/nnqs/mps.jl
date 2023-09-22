

struct MPS{T<:Number} <: AbstractNNQS
	data::Vector{Array{T, 3}}
end

Flux.@functor MPS 

function MPS(f, ::Type{T}, physpaces::Vector{Int}, virtualpaces::Vector{Int}) where {T <: Number}
	L = length(physpaces)
	(length(virtualpaces) == L+1) || throw(DimensionMismatch())
	any(virtualpaces .== 0) &&  throw(ArgumentError("virtualpace can not be 0"))
	mpstensors = [f(T, virtualpaces[i], physpaces[i], virtualpaces[i+1]) for i in 1:L]
	return MPS(mpstensors)
end

Base.length(x::MPS) = length(x.data)
Base.eltype(::Type{MPS{T}}) where T = T
Base.eltype(x::MPS) = eltype(typeof(x))

function MPS(::Type{T}, physpaces::Vector{Int}; D::Int) where {T<:Number}
	virtualpaces = max_bond_dimensions(physpaces, D)
	return MPS(randn, T, physpaces, virtualpaces)
end
MPS(physpaces::Vector{Int}; D::Int) = MPS(Float64, physpaces; D=D)
MPS(::Type{T}, L::Int; D::Int, d::Int=2) where {T <: Number} = MPS(T, [d for i in 1:L]; D=D)
MPS(L::Int; D::Int, d::Int=2) = MPS(Float64, L; d=d, D=D)


function _Ψ_util(m::MPS, x::ComputationBasis)
	@assert length(m) == length(x)
	v = m.data[1][:, x[1], :]	
	for i in 2:length(x)
		v = v * m.data[i][:, x[i], :]
	end
	return only(v)
end
_Ψ(m::MPS, x::ComputationBasis) = _Ψ_util(m, dropgrad(_state_to_index(x)))

sys_size(x::MPS) = length(x)
Base.copy(x::MPS) = MPS(copy(x.data))

_state_to_index(x::ComputationBasis) = [(item == -1) ? 2 : 1 for item in x]
# Zygote.@adjoint state_to_index(x::ComputationBasis) = state_to_index(x), z -> (nothing,)

function max_bond_dimensions(physpaces::Vector{Int}, D::Int) 
	L = length(physpaces)
	left = 1
	right = 1
	virtualpaces = Vector{Int}(undef, L+1)
	virtualpaces[1] = left
	for i in 2:L
		virtualpaces[i] = min(virtualpaces[i-1] * physpaces[i-1], D)
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = min(virtualpaces[i], physpaces[i] * virtualpaces[i+1])
	end
	return virtualpaces
end

bond_dimension(psi::MPS, bond::Int) = begin
	((bond >= 1) && (bond <= length(psi))) || throw(BoundsError())
	size(psi.data[bond], 3)
end 
bond_dimensions(psi::MPS) = [bond_dimension(psi, i) for i in 1:length(psi)]
bond_dimension(psi::MPS) = maximum(bond_dimensions(psi))


# prepare MPS in right-canonical form
function rightorth!(psi0::MPS; normalize::Bool=true)
	psi = psi0.data
	L = length(psi)
	workspace = eltype(psi0)[]
	for i in L:-1:2
		l, q = tlq!(psi[i], (1,), (2, 3), workspace)
		normalize && LinearAlgebra.normalize!(l)
		psi[i] = q
		psi[i-1] = reshape(tie(psi[i-1], (2, 1)) * l, size(psi[i-1], 1), size(psi[i-1], 2), size(l, 2))
	end
	normalize && LinearAlgebra.normalize!(psi[1])
	return psi0
end
rightorth(psi::MPS; kwargs...) = rightorth!(copy(psi); kwargs...)

isrightcanonical(a::MPS; kwargs...) = all(x->isrightcanonical(x; kwargs...), a.data)
function isrightcanonical(psij::AbstractArray{<:Number, 3}; kwargs...)
	m2 = tie(psij, (1,2))
	r = m2 * m2'
	return isapprox(r, one(r); kwargs...) 
end

# prepare MPS in left-canonical form
function leftorth!(psi0::MPS; normalize::Bool=true)
	psi = psi0.data
	L = length(psi)
	workspace = eltype(psi0)[]
	for i in 1:L-1
		q, r = tqr!(psi[i], (1, 2), (3,), workspace)
		normalize && LinearAlgebra.normalize!(r)
		psi[i] = q
		psi[i+1] = reshape(r * tie(psi[i+1], (1, 2)), size(r, 1), size(psi[i+1], 2), size(psi[i+1], 3))
	end
	normalize && LinearAlgebra.normalize!(psi[L])
	return psi0
end
leftorth(psi::MPS; kwargs...) = leftorth!(copy(psi); kwargs...)
isleftcanonical(a::MPS; kwargs...) = all(x->isleftcanonical(x; kwargs...), a.data)
function isleftcanonical(psij::AbstractArray{<:Number, 3}; kwargs...)
	m2 = tie(psij, (2, 1))
	r = m2' * m2
	return isapprox(r, one(r); kwargs...) 
end

function increase_bond!(psi::MPS; D::Int)
	if bond_dimension(psi) < D
		virtualpaces = max_bond_dimensions([2 for i in 1:length(psi)], D)
		for i in 1:length(psi)
			sl = max(min(virtualpaces[i], D), size(psi.data[i], 1))
			sr = max(min(virtualpaces[i+1], D), size(psi.data[i], 3))
			m = zeros(eltype(psi), sl, size(psi.data[i], 2), sr)
			m[1:size(psi.data[i], 1), :, 1:size(psi.data[i], 3)] .= psi.data[i]
			psi.data[i] = m
		end
	end
	return psi
end


# lq decomposition
function tlq!(a::StridedMatrix) 
    l, q = LinearAlgebra.lq!(a)
    return l, Matrix(q)
end
function tlq!(a::AbstractArray{T, N}, left::NTuple{N1, Int}, right::NTuple{N2, Int}, workspace::AbstractVector{T}=similar(a, length(a))) where {T<:Number, N, N1, N2}
    (N == N1 + N2) || throw(DimensionMismatch())
	if length(workspace) <= length(a)
		resize!(workspace, length(a))
	end
    newindex = (left..., right...)
    a1 = permute(a, newindex)
    shape_a = size(a1)
    dimu = shape_a[1:N1]
    s1 = prod(dimu)
    dimv = shape_a[(N1+1):end]
    s2 = prod(dimv)
    bmat = copyto!(reshape(view(workspace, 1:length(a)), s1, s2), reshape(a1, s1, s2))
    # F = LinearAlgebra.lq!(bmat)
    # u = Matrix(F.L)
    # v = Matrix(F.Q)
    u, v = tlq!(bmat)
    s = size(v, 1)
    return reshape(u, dimu..., s), reshape(v, s, dimv...)
end
permute(m::AbstractArray, perm) = PermutedDimsArray(m, perm)

function _group_extent(extent::NTuple{N, Int}, idx::NTuple{N1, Int}) where {N, N1}
    ext = Vector{Int}(undef, N1)
    l = 0
    for i=1:N1
        ext[i] = prod(extent[(l+1):(l+idx[i])])
        l += idx[i]
    end
    return NTuple{N1, Int}(ext)
end


function tie(a::AbstractArray{T, N}, axs::NTuple{N1, Int}) where {T, N, N1}
    (sum(axs) != N) && error("total number of axes should equal to tensor rank.")
    return reshape(a, _group_extent(size(a), axs))
end
