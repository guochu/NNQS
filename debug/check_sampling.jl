# push!(LOAD_PATH, "src")

include("src/includes.jl")

using Random
# using NNQS
using Flux, Flux.Optimise
using StatsBase, LinearAlgebra


function main1(nsamples)
	h = IsingChain(h=1, J=1)
	n_visible = 4
	Random.seed!(1234)
	mps = MPS(Float64, n_visible, D=2)
	rightorth!(mps)
	# nsamples = 100000
	sampler = MetropolisLocal(n_visible, n_thermal=10000, n_sample_per_chain=nsamples, n_discard=100)	
	samples, counts = generate_samples(h, sampler, mps)
	r = Dict(samples[:, i]=>counts[i] for i in 1:length(counts))

	# pr = Dict(v / nsamples for (k, v) in r)

	v1 = Float64[]
	v2 = Float64[]
	for (k, v) in r
		p_act = v / nsamples
		p_tar = abs2(Ψ(mps, k))
		println(k, " ", p_tar, " ", p_act)
		push!(v1, p_tar)
		push!(v2, p_act)
	end

	println("average distance is ", norm(v1 - v2))
end

function main2(nsamples)
	h = IsingChain(h=1, J=1)
	n_visible = 4
	Random.seed!(1234)
	mps = MPS(Float64, n_visible, D=2)
	rightorth!(mps)
	# nsamples = 100000
	sampler = AutoRegressiveSampler(n_visible, n_sample_per_chain=nsamples)	
	samples, counts = generate_samples(h, sampler, mps)
	r = Dict(samples[:, i]=>counts[i] for i in 1:length(counts))

	# pr = Dict(v / nsamples for (k, v) in r)

	v1 = Float64[]
	v2 = Float64[]
	for (k, v) in r
		p_act = v / nsamples
		p_tar = abs2(Ψ(mps, k))
		println(k, " ", p_tar, " ", p_act)
		push!(v1, p_tar)
		push!(v2, p_act)
	end

	println("average distance is ", norm(v1 - v2))
end

function main3(nsamples)
	h = IsingChain(h=1, J=1)
	n_visible = 4
	Random.seed!(1234)
	mps = MPS(Float64, n_visible, D=2)
	rightorth!(mps)
	# nsamples = 100000
	sampler = BatchAutoRegressiveSampler(n_visible, n_sample_per_chain=nsamples)	
	samples, counts = generate_samples(h, sampler, mps)
	println(counts)
	r = Dict(samples[:, i]=>counts[i] for i in 1:length(counts))

	# pr = Dict(v / nsamples for (k, v) in r)

	v1 = Float64[]
	v2 = Float64[]
	for (k, v) in r
		p_act = v / nsamples
		p_tar = abs2(Ψ(mps, k))
		println(k, " ", p_tar, " ", p_act)
		push!(v1, p_tar)
		push!(v2, p_act)
	end

	println("average distance is ", norm(v1 - v2))
end

