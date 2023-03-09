include("./dev_mol_hamiltonian.jl")

using Random
using Logging

import NNQS
using NNQS: log

using Statistics
const libcouple = "libcouple.so"

const datatype = Int8
const coeff_dtype = Float64

struct MolecularHamiltonianInt{T} <: Hamiltonian where T <:Integer
    n_qubits::Int
    pauli_mat12_dict::Dict{Int, Vector{T}}
    pauli_mat23_dict::Dict{Int, Matrix{T}}
    coeffs_dict::Dict{Int, Vector{Float64}}
end

struct MolecularHamiltonianIntOpt{T} <: Hamiltonian where T <:Integer
    n_qubits::Int
    n_ele_max::Int
    idxs::Vector{Int64}
    pauli_mat12::Matrix{T}
    pauli_mat23::Matrix{T}
    coeffs::Vector{coeff_dtype}
end

struct MolecularHamiltonianBit <: Hamiltonian
    n_qubits::Int
    pauli_mat12_dict::Dict{Int, BitArray}
    pauli_mat23_dict::Dict{Int, BitMatrix}
    coeffs_dict::Dict{Int, Vector{Float64}}
end


diagonal_coupling(h::MolecularHamiltonianInt, state::NNQS.ComputationBasis) = 0.
diagonal_coupling(h::MolecularHamiltonianBit, state::NNQS.ComputationBasis) = 0.
diagonal_coupling(h::MolecularHamiltonianIntOpt, state::NNQS.ComputationBasis) = 0.
# diagonal_coupling(h::MolecularHamiltonianInt, state::AbstractVector{<:Integer}) = 0.
# diagonal_coupling(h::MolecularHamiltonianBit, state::AbstractVector{<:Integer}) = 0.

@inline function sum_1_loop(x::Vector{datatype})
    s::datatype = 0
    for i in x
        s += i
    end
    return s
end

function _state2id(state::AbstractArray{datatype})
    two = 1
    res = 0
    for i in state
        if i == 1
            res += two
        end
        two *= 2
    end
    return res
end

function extract_indices_ham_int(ham::MolecularHamiltonian, eps::Float64=1e-7)
    log("extract_indices_ham_int: n_qubit = $(ham.n_qubits), n_qubit_ops = $(length(ham.qubit_op))")
    N = ham.n_qubits
    K = length(ham.qubit_op)

    
    # only 0/1
    global datatype
    pauli_mat12 = zeros(datatype, N, K)
    pauli_mat23 = zeros(datatype, N, K)
    
    # qubits <= 64, key::Int64
    pauli_mat12_dict = Dict{Int, Vector{datatype}}()
    pauli_mat23_dict = Dict{Int, Vector{Vector{datatype}}}()
    coeffs_dict = Dict{Int, Vector{Float64}}()
    
    row_idx::Int32 = 1
    cnt::Int8 = 0
    # for (term, coeff::Float64) in ham.qubit_op
    for (term, coeff) in ham.qubit_op
        # abs(imag(coeff)) > 1e-14 && error("Only support real-valued Hamiltonian!")
        abs(imag(coeff)) > eps && error("Only support real-valued Hamiltonian!")
        cnt = 0
        for (pos, pauli) in term
            if pauli == "X"
                pauli_mat12[pos+1, row_idx] = 1
            elseif pauli == "Y"
                pauli_mat12[pos+1, row_idx] = 1
                pauli_mat23[pos+1, row_idx] = 1
                cnt += 1
            else
                pauli_mat23[pos+1, row_idx] = 1
            end
        end


        # fuse const calculation
        coeff = real(coeff) * real((-1im)^cnt)
        
        sid::Int = _state2id(pauli_mat12[:,row_idx])
        if haskey(coeffs_dict, sid)
            append!(coeffs_dict[sid], coeff)
            append!(pauli_mat23_dict[sid], [pauli_mat23[:,row_idx]])
        else
            pauli_mat12_dict[sid] = pauli_mat12[:,row_idx]
            pauli_mat23_dict[sid] = [pauli_mat23[:,row_idx]]
            coeffs_dict[sid] = Vector{Float64}([coeff])
        end
        row_idx += 1
    end

    
    # Improve cache utilization: pack memory into new continuous memory buf
    # new continuous buf 
    pauli_mat12_buf = zeros(datatype, N, pauli_mat12_dict.count)
    pauli_mat23_buf = zeros(datatype, N, K)
    coeffs_buf = zeros(Float64, K)
    # final result
    pauli_mat12_pack_dict = Dict{Int, Vector{datatype}}()
    pauli_mat23_pack_dict = Dict{Int, Matrix{datatype}}()
    coeffs_pack_dict = Dict{Int, Vector{Float64}}()
    
    num_idx = 0
    i = 1
    for (sid, pm23) in pauli_mat23_dict
        num = size(pm23, 1)
        for (i, x) in enumerate(pm23)
            # copy!(view(pauli_mat23_buf, :, num_idx+i), x)
            @. pauli_mat23_buf[:,num_idx+i] = x
        end
        pauli_mat23_pack_dict[sid] = view(pauli_mat23_buf, :, num_idx+1:num_idx+num)
        
        @. coeffs_buf[num_idx+1:num_idx+num] = coeffs_dict[sid]
        coeffs_pack_dict[sid] = view(coeffs_buf, num_idx+1:num_idx+num)
        
        @. pauli_mat12_buf[:, i] = pauli_mat12_dict[sid]
        pauli_mat12_pack_dict[sid] = view(pauli_mat12_buf, :, i)
        
        i += 1
        num_idx += num
    end
    h = MolecularHamiltonianInt{datatype}(N, pauli_mat12_pack_dict, pauli_mat23_pack_dict, coeffs_pack_dict)
    return h
end

function extract_indices_ham_int_opt(ham::MolecularHamiltonian, eps::Float64=1e-7)
    log("extract_indices_ham_int_opt: n_qubit = $(ham.n_qubits), n_qubit_ops = $(length(ham.qubit_op))")
    N = ham.n_qubits
    K = length(ham.qubit_op)
    
    # only 0/1
    global datatype
    pauli_mat12 = zeros(datatype, N, K)
    pauli_mat23 = zeros(datatype, N, K)
    
    # qubits <= 64, key::Int64
    pauli_mat12_dict = Dict{Int, Vector{datatype}}()
    pauli_mat23_dict = Dict{Int, Vector{Vector{datatype}}}()
    coeffs_dict = Dict{Int, Vector{coeff_dtype}}()

    row_idx::Int32 = 1
    cnt::Int8 = 0
    for (term, coeff) in ham.qubit_op
        abs(imag(coeff)) > eps && error("Only support real-valued Hamiltonian!")
        cnt = 0
        for (pos, pauli) in term
            if pauli == "X"
                pauli_mat12[pos+1, row_idx] = 1
            elseif pauli == "Y"
                pauli_mat12[pos+1, row_idx] = 1
                pauli_mat23[pos+1, row_idx] = 1
                cnt += 1
            else
                pauli_mat23[pos+1, row_idx] = 1
            end
        end



        # fuse const calculation
        coeff = real(coeff) * real((-1im)^cnt)
        
        sid::Int = _state2id(pauli_mat12[:,row_idx])

        if haskey(coeffs_dict, sid)
            append!(coeffs_dict[sid], coeff)
            append!(pauli_mat23_dict[sid], [pauli_mat23[:,row_idx]])
        else
            pauli_mat12_dict[sid] = pauli_mat12[:,row_idx]
            pauli_mat23_dict[sid] = [pauli_mat23[:,row_idx]]
            coeffs_dict[sid] = Vector{coeff_dtype}([coeff])
        end
        row_idx += 1
    end
    
    # Improve cache utilization: pack memory into new continuous memory buf
    # new continuous buf 
    pauli_mat12_buf = zeros(datatype, N, pauli_mat12_dict.count)
    pauli_mat23_buf = zeros(datatype, N, K)
    coeffs_buf = zeros(coeff_dtype, K)
    # final result
    idxs = zeros(Int64, pauli_mat12_dict.count+1)

    num_idx, i = 0, 1
    for (sid, pm23) in pauli_mat23_dict
        num = size(pm23, 1)

        for (i, x) in enumerate(pm23)
            # copy!(view(pauli_mat23_buf, :, num_idx+i), x)
            @. pauli_mat23_buf[:,num_idx+i] = x
        end


        @. coeffs_buf[num_idx+1:num_idx+num] = coeffs_dict[sid]
        @. pauli_mat12_buf[:, i] = pauli_mat12_dict[sid]
        
        i += 1
        num_idx += num
        idxs[i] = num_idx
    end
    #println("-------------------------debug info-----------------------")
    # println(pauli_mat12_dict)
    # println(pauli_mat23_dict)
    #println("debug_info")
    h = MolecularHamiltonianIntOpt{datatype}(N, pauli_mat12_dict.count, idxs, pauli_mat12_buf, pauli_mat23_buf, coeffs_buf)
    ptr2idx = pointer(idxs)
    ptr2coeff = pointer(coeffs_buf)
    v_12 = vec(h.pauli_mat12)
    ptr2v12 = pointer(v_12)
    v_23 = vec(h.pauli_mat23)
    ptr2v23 = pointer(v_23)
    dim1_12 = size(pauli_mat12_buf, 1)
    dim2_12 = size(pauli_mat12_buf, 2)
    dim1_23 = size(pauli_mat23_buf, 1)
    dim2_23 = size(pauli_mat23_buf, 2)
    idx_len = length(idxs)
    coeff_len = length(coeffs_buf)

    #println("C test here")
    ccall((:Hamtion_buffer, libcouple), Cvoid, (Cint, Cint, Ptr{Cint}, Ptr{Cint}, Cint ,Cint, Ptr{Cint}, Cint, Cint, Ptr{Cdouble}, Cint, Cint),
         N, pauli_mat12_dict.count, ptr2idx, ptr2v12, dim1_12, dim2_12, ptr2v23, dim1_23, dim2_23, ptr2coeff, idx_len, coeff_len)
    
    #println("-------------------------debug info-----------------------")
    #println(h)

    return h
end

new_bitarray(x::AbstractArray) = BitArray(ntuple(i->(x[i]==1), length(x)))
aligned(x)::Int64 = (UInt64(x)+63)&(~UInt64(63))

function extract_indices_ham_bit(ham::MolecularHamiltonian, eps::Float64=1e-7)
    log("extract_indices_ham_bit: n_qubit = $(ham.n_qubits), n_qubit_ops = $(length(ham.qubit_op))")
    global datatype
    N = ham.n_qubits
    K = length(ham.qubit_op)
    N64x = aligned(N)

    # only 0/1
    pauli_mat12 = zeros(datatype, N, K)
    pauli_mat23 = zeros(datatype, N, K)
    
    # qubits <= 64, key::Int64
    pauli_mat12_dict = Dict{Int, Vector{datatype}}()
    pauli_mat23_dict = Dict{Int, Vector{Vector{datatype}}}()
    coeffs_dict = Dict{Int, Vector{Float64}}()
    
    row_idx::Int32 = 1
    cnt::Int8 = 0
    # Parse Hamiltonian operator and store key information in the temp Dict
    # for (term, coeff::Float64) in ham.qubit_op
    for (term, coeff) in ham.qubit_op
        abs(imag(coeff)) > eps && error("Only support real-valued Hamiltonian!")
        cnt = 0
        
        for (pos, pauli) in term
            if pauli == "X"
                pauli_mat12[pos+1, row_idx] = 1
            elseif pauli == "Y"
                pauli_mat12[pos+1, row_idx] = 1
                pauli_mat23[pos+1, row_idx] = 1
                cnt += 1
            else
                pauli_mat23[pos+1, row_idx] = 1
            end
        end
        # fuse const calculation
        coeff = real(coeff) * real((-1im)^cnt)
        
        sid::Int = _state2id(pauli_mat12[:,row_idx])
        if haskey(coeffs_dict, sid)
            append!(coeffs_dict[sid], coeff)
            append!(pauli_mat23_dict[sid], [pauli_mat23[:,row_idx]])
        else
            pauli_mat12_dict[sid] = pauli_mat12[:,row_idx]
            pauli_mat23_dict[sid] = [pauli_mat23[:,row_idx]]
            coeffs_dict[sid] = Vector{Float64}([coeff])
        end
        row_idx += 1
    end
    
    # Improve cache utilization: pack memory into new continuous memory buf (BitArray)
    # new continuous buf 
    pauli_mat12_buf = falses(N64x, pauli_mat12_dict.count)
    pauli_mat23_buf = falses(N64x, K)
    coeffs_buf = zeros(Float64, K)
    # final result
    pauli_mat12_pack_dict = Dict{Int, BitArray}()
    pauli_mat23_pack_dict = Dict{Int, BitMatrix}()
    coeffs_pack_dict = Dict{Int, Vector{Float64}}()
    
    num_idx = 0
    i = 1
    for (sid, pm23) in pauli_mat23_dict
        num = size(pm23, 1)
        for (i, x) in enumerate(pm23)
            # copy!(view(pauli_mat23_buf, :, num_idx+i), x)
            # @. pauli_mat23_buf[:,num_idx+i] = x
            _tpl = ntuple(i->(x[i]==1), length(x))
            @. pauli_mat23_buf[1:N, num_idx+i] = _tpl
        end
        pauli_mat23_pack_dict[sid] = view(pauli_mat23_buf, :, num_idx+1:num_idx+num)
        
        @. coeffs_buf[num_idx+1:num_idx+num] = coeffs_dict[sid]
        coeffs_pack_dict[sid] = view(coeffs_buf, num_idx+1:num_idx+num)
        
        _tpl = ntuple(i->(pauli_mat12_dict[sid][i]==1), length(pauli_mat12_dict[sid]))
        @. pauli_mat12_buf[1:N, i] = _tpl
        pauli_mat12_pack_dict[sid] = view(pauli_mat12_buf, :, i)
        
        i += 1
        num_idx += num
    end

    h = MolecularHamiltonianBit(N, pauli_mat12_pack_dict, pauli_mat23_pack_dict, coeffs_pack_dict)
    return h
end

# Qubit state storage: [U]Int64/32/16/8
function coupled_states(h::MolecularHamiltonianInt, _state::NNQS.ComputationBasis, eps::Float64=1e-14)
    global datatype    
    N = h.n_qubits

    state = ones(datatype, length(_state))
    for (i, v) in enumerate(_state)
        (v == -1) && (state[i] = 0)
        # (v == 0) && (state[i] = 0)
    end
    # @. state[_state == -1] = 0

    nstate = zeros(datatype, N)
    _tstate = zeros(datatype, N)
    # eps = 1e-14

    res_states = ones(Int, N, h.pauli_mat12_dict.count)
    res_coefs = zeros(Float64, h.coeffs_dict.count)
    res_cnt = 0
    for (sid, pm12) in h.pauli_mat12_dict
        coef = 0.0
        @inbounds for i in 1:size(h.pauli_mat23_dict[sid], 2)
            x = view(h.pauli_mat23_dict[sid], :, i)
            @. _tstate = x & state
            coef += ((-1)^(sum_1_loop(_tstate))) * h.coeffs_dict[sid][i]
        end

        abs(coef) < eps && continue 
        res_cnt += 1
        res_coefs[res_cnt] = coef

        @. nstate = xor(state, pm12)
        for (i, vs) in enumerate(nstate)
            (vs == 0) && (res_states[i, res_cnt] = -1)
        end
        # for (i, vs) in enumerate(nstate)
        #     (vs == 0) && (res_states[i, res_cnt] = 0)
        # end
        # @. res_states[nstate==0, res_cnt] = -1
    end

    _states, _coefs = view(res_states, :, 1:res_cnt), view(res_coefs, 1:res_cnt)
    return _states, _coefs
end

# Qubit state storage: [U]Int64/32/16/8
function coupled_states(h::MolecularHamiltonianIntOpt, _state::NNQS.ComputationBasis, eps::Float64=1e-14)
    global datatype    
    N = h.n_qubits
    
    target_value = -1 # 0,1/+1,-1 represent the qubit

    state = ones(datatype, length(_state))
    for (i, v) in enumerate(_state)
        (v == target_value) && (state[i] = 0)
    end
    # @. state[_state == -1] = 0

    nstate = zeros(datatype, N)
    _tstate = zeros(datatype, N)

    res_states = ones(Int, N, h.n_ele_max)
    res_coefs = zeros(coeff_dtype, h.n_ele_max)
    res_cnt = 0

    for sid in 1:size(h.pauli_mat12, 2)
        coef = 0.0
        
        st, ed = h.idxs[sid]+1, h.idxs[sid+1]
        @inbounds for i in st:ed
            x = view(h.pauli_mat23, :, i)
            @. _tstate = x & state
            coef += ((-1)^(sum_1_loop(_tstate))) * h.coeffs[i]
        end
        
        #println("float", coef, eps)
        abs(coef) < eps && continue 
        res_cnt += 1
        res_coefs[res_cnt] = coef
        
        pm12 = view(h.pauli_mat12, :, sid)
        @. nstate = xor(state, pm12)

        for (i, vs) in enumerate(nstate)
            (vs == 0) && (res_states[i, res_cnt] = target_value)
        end

        # @. res_states[nstate==0, res_cnt] = -1
    end

    # println("couple debuf info route time =", res_cnt)
    # println("martrix is: ", h.pauli_mat12)
    # println("martrix dim is: ", size(h.pauli_mat12, 2))


    _states, _coefs = view(res_states, :, 1:res_cnt), view(res_coefs, 1:res_cnt)
    return _states, _coefs
end

# Qubit state storage: BitArray
function coupled_states(h::MolecularHamiltonianBit, _state::NNQS.ComputationBasis, eps::Float64=1e-14)
    # the number of qubits in a state
    N = h.n_qubits
    # align N upper to 64x, for improving bit operator performance
    N64x = aligned(N)
    
    # preallocate memory and transfer [-1,1,1,-1...] to [0,1,1,0,...]
    state = falses(N64x)
    state[1:N] .= new_bitarray(_state)
    
    # the number of UInt64 consumed by a qubit state
    NUInt64 = length(state.chunks)
    nstate = falses(N64x)
    # eps = 1e-14

    # preallocate maximum result/output memory
    res_states = ones(Int, N, h.pauli_mat12_dict.count)
    res_coefs = zeros(Float64, h.coeffs_dict.count)
    res_cnt = 0
    for (sid, pm12) in h.pauli_mat12_dict
        # calculation of coef of Hamiltonian term
        coef = 0.0
        # @assert size(pauli_mat23_dict[sid], 2) == length(pauli_mat23_dict[sid].chunks)
        i_chk = 0
        @inbounds for i in 1:size(h.pauli_mat23_dict[sid], 2)
            cnt_1 = 0
            for j = 1:NUInt64
                x = h.pauli_mat23_dict[sid].chunks[i_chk+j]
                cnt_1 += count_ones(x & state.chunks[j])
            end
            coef += ((-1)^(cnt_1)) * h.coeffs_dict[sid][i]
            i_chk += NUInt64
        end
        
        # discard unless result
        abs(coef) < eps && continue 
        res_cnt += 1
        res_coefs[res_cnt] = coef

        # calculation of result state of this term
        for j = 1:NUInt64
            @inbounds nstate.chunks[j] = xor(state.chunks[j], pm12.chunks[j])
        end
    
        # # transfer result state [0,1,1,0,...] to [-1,1,1,-1...]
        for (i, vs) in enumerate(view(nstate, 1:N))
            @inbounds (vs == 0) && (res_states[i, res_cnt] = -1)
        end
        # transfer result state [0,1,1,0,...] to [0,1,1,0...]
        # for (i, vs) in enumerate(view(nstate, 1:N))
        #     @inbounds (vs == 0) && (res_states[i, res_cnt] = 0)
        # end
    end

    # reduce memory copy
    _states, _coefs = view(res_states, :, 1:res_cnt), view(res_coefs, 1:res_cnt)
    return _states, _coefs
end

function check_results(molecule, data_dir_prefix="mol_ham_data", storage_type="Int_opt")
    # log("molecule= $molecule")

    # config your data path
    data_path = "$data_dir_prefix/$molecule/" * "qubit_op.data"
    ham, h = get_ham(data_path, storage_type)
    # log("n_qubit_ops: $(length(ham.qubit_op))")


    # random get state
    N = ham.n_qubits
    M = h.n_ele_max
	Random.seed!(10)
    flag1 = M * N
    flag2 = M

    # epoches = 200
    epoches = 5
    states = rand(-1:2:2, N, epoches)
    # res_states_batch, res_coefs_batch, res_cnt_batch = coupled_states(h, states)


    st = time_ns()
    for i in 1:epoches
        state = states[:, i]
        if i == 1
        end
        states1, coefs1 = coupled_states(ham, state)
        # println("-------------debug info----------------")
        # println(states1)
        # println(coefs1)
        st_len = length(state)
        cstate = zeros(Int64, flag1)
        ccoffe = zeros(Float64, flag2)
        res_cnt_tmp = zeros(Int64, 1)
        #statesv2, coefsv2 = res_states_batch[:, 1:res_cnt_batch[i], i], res_coefs_batch[1:res_cnt_batch[i], i]
        ccall((:coupled_states, libcouple), Cvoid, (Ptr{Clonglong}, Cdouble, Cint, Ptr{Clonglong}, Ptr{Cdouble}, Ptr{Clonglong}), state, 1e-14, st_len, cstate, ccoffe, res_cnt_tmp)
        res_cnt = res_cnt_tmp[1]
        cstate = cstate[1:N*res_cnt]
        statesv2 = reshape(cstate, N, res_cnt)
        coefsv2 = view(ccoffe, 1:res_cnt)
        

        #states2 = reshape(cstate, size(states1))
        #coefsv2 = ccoffe
        #statesv2, coefsv2 = coupled_states(h, state)
        # println("$(res_cnt_batch[i]), coefs1: $(size(coefs1)) v2: $(size(coefsv2)) type: $(typeof(coefs1)) $(typeof(coefsv2))")
        s1 = sort(coefs1)
        # println("s1: $(s1[1:10])")
        s2 = sort(coefsv2)
        # println("s2: $(s2[1:10])")
        mean_err, max_err = mean(abs.(s1-s2)), maximum(abs.(s1-s2))
        @assert mean_err < 1e-8 && max_err < 1e-7 "mean_err: $mean_err, max_err: $max_err"
        # @assert sort(coefs1) == sort(coefsv2) "$(state)\n$(sort(coefs1)[1:10])\n$(sort(coefsv2)[1:10])"
        @assert sort(states1, dims=2) == sort(statesv2, dims=2) "$(state); $(states1); $(statesv2)"
    end
    elapse_time = (time_ns() - st) / 10^9
    log("[PASS $molecule] Timing of coupled_states: $elapse_time s\n")
end

function test_perf(molecule, epoches=200, msg=1, data_dir_prefix="mol_ham_data", storage_type="Int_opt")
    # config your data path
    data_path = "$data_dir_prefix/$molecule/" * "qubit_op.data"
    ham, h = get_ham(data_path, storage_type)

    # random get state
    N = ham.n_qubits
	Random.seed!(1)
    
    st = time_ns()
    state = rand(-1:2:2, N, epoches)
    for i in 1:epoches
        states, coefs = coupled_states(h, state[:, i])
    end
    elapse_time = (time_ns() - st) / 10^9
    log("[PERF $molecule] $(epoches) epoches; \t coupled_states time: $elapse_time s")
end

function test_drive(ptype, data_dir_prefix="mol_ham_data")
    molecules = ["h2", "lih", "h2o", "c2", "n2", "nh3", "li2o", "c2h4o", "c3h6", "c2h4o2"]

    molecule = "h2"
    #molecule = "c2"
    # molecule = "nh3"
    # molecule = "co2"
    #molecule = "li2o"
    # molecule = "c2h4o"
    storage_type = "Int_opt"
    # data_dir_prefix = "mol_ham_data"
    data_dir_prefix = "mol_ham_data"

    if ptype == 1
        check_results(molecule, data_dir_prefix, storage_type)
        log(">>CHECK PASS<<< $molecule")
    elseif ptype == 2
        for m in molecules
            check_results(m, data_dir_prefix, storage_type)
        end
        log(">>>CHECK PASS<<< $molecules")
    else
        n_rpt = 6
        # epoches = 500
        epoches = 10000
        if ptype == 3
            fmsg = 1
            @time "==Total Time== ($epoches * $n_rpt) " for i in 1:n_rpt
                @time "[$i-th]" test_perf(molecule, epoches, fmsg, data_dir_prefix, storage_type)
                fmsg = 0
            end
        else
            for m in molecules
                fmsg = 1
                @time "==Total Time== ($epoches * $n_rpt)" for i in 1:n_rpt
                    @time "[$i-th]" test_perf(m, epoches, fmsg, data_dir_prefix, storage_type)
                    println("\n")
                    fmsg = 0
                end
                println("\n")
            end
        end
        log("\n>>>Perf Over<<< ^v^")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_drive(parse(Int, ARGS[1]))
    # data_path = "mol_ham_data/h2/qubit_op.data"
    # ham, h = get_ham(data_path, "Int_opt")



    # N = ham.n_qubits
    # M = h.n_ele_max


	# Random.seed!(1)
    
    # st = time_ns()
    # #state = rand(-1:2:2, N, 200)
    # state = [1, 1, -1, -1]
    # state2ptr = pointer(state)
    # st_len = length(state)




    
    

    # flag1 = M * N
    # flag2 = M



    # cstate = zeros(Int64, flag1)
    # ccoffe = zeros(Float64, flag2)

    # ccall((:coupled_states, libcouple), Cvoid, (Ptr{Clonglong}, Cdouble, Cint, Ptr{Clonglong}, Ptr{Cdouble}), state[:,1], 0.00000001, st_len, cstate, ccoffe)
    # flush(stdout)

    # println(cstate)
    # println(ccoffe)

    





    # states, coefs = coupled_states(h, state[:, 1])
    # println(states)
    # println(coefs)

    # # @ccall couplelib.read_qubit_op(data_path)
    # # ccall((:read_qubit_op, "libcouple"), Cvoid, (Cstring), datapath)
    # # ccall((:read_qubit_op, "libcouple"), Cvoid, (Cstring, Ptr), datapath)

end




