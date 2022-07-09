

function _collect_parameters!(v::AbstractVector, p::AbstractArray{<:Number})
	for x in p
		push!(v, x)
	end
	return v
end

function _collect_parameters!(v::AbstractVector, p::AbstractDict)
	for (k, x) in p
		_collect_parameters!(v, x)
	end
	return v
end


function parameters(x::Params)
	v = Number[]
	for item in x
		_collect_parameters!(v, item)
	end
	return [v...]
end

function parameters(x::Grads)
	v = Number[]
	for p in x.params
		_collect_parameters!(v, x[p])
	end
	return [v...]
end

function _reset_parameters_util!(x::AbstractArray{<:Number}, v::AbstractVector, pos::Int)
	L = length(x)
	pos_f = pos + L
	@assert (pos_f <= length(v))
	x[:] .= view(v, pos+1:pos_f)
	return pos_f
end

function _reset_parameters_util!(x::AbstractDict, v::AbstractVector, pos::Int)
	for (k, item) in x
		pos = _reset_parameters_util!(item, v, pos)
	end
	return pos
end

function reset!(p::Params, v::AbstractVector)
	pos = 0
	for item in p
		pos = _reset_parameters_util!(item, v, pos)
	end
	(pos == length(v)) || error("number of parameters mismatch.")
end

