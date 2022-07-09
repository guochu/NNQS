# QCQC: https://gitlab.com/auroraustc/qcqc.git
# include("./qcqc/quantum_spins_helper.jl")
# or just copy the read_binary_qubit_op() and count_qubits() function from quantum_spins_helper.jl
push!(LOAD_PATH, "../src")

using LinearAlgebra

using NNQS
import NNQS: coupled_states, diagonal_coupling

# Copied from QCQC
function count_qubits(qubit_op::Tuple)
    n_qubits = 1
    for i in qubit_op
        target_indices = vcat([0], Vector{Int64}([j[1] for j in i[1]]))
        n_qubits = max(n_qubits, maximum(target_indices) + 1)
    end
    return n_qubits
end

function readn(io::IO, T, num_vals::Int64)
    buffer_array = Vector{T}()
    for i in 1:num_vals
        try
            tmp_val = read(io, T)
            push!(buffer_array, tmp_val)
        catch EOFError
            break
        end
    end
    return buffer_array
end

function readn(io::IO, T, num_vals::Int32)
    return readn(io::IO, T, Int64(num_vals))
end

function read_binary_qubit_op(filename::String)
    #=
    Read the saved binary format of QubitOperator saved by QCQC's
    utils.save_binary_qubit_op()

    Notes:
        Make sure that the magic number is the same as QCQC.
    =#
    magic_number = Float64(11.2552)
    f = open(filename, "r")
    identifier = readn(f, Float64, 1)[1]
    err_msg = identifier != magic_number && error("The file is note saved by QCQC.")

    n_qubits = readn(f, Int32, 1)[1]
    pauli_symbol_dict = Dict(
        0 => "I",
        1 => "X",
        2 => "Y",
        3 => "Z"
    )
    qubit_op_dict = Dict()
    coeff_tmp = readn(f, ComplexF64, 1)
    pauli_str_tmp = readn(f, Int32, n_qubits)
    while length(coeff_tmp) != 0 && length(pauli_str_tmp) != 0
        err_msg = length(pauli_str_tmp) != n_qubits && error("The file seems damaged.")
        pauli_str_tuple = Tuple([(i - 1, pauli_symbol_dict[pauli_str_tmp[i]])
                                 for i in 1:n_qubits
                                 if pauli_str_tmp[i] != 0])
        qubit_op_dict[pauli_str_tuple] = coeff_tmp[1]
        coeff_tmp = readn(f, ComplexF64, 1)
        pauli_str_tmp = readn(f, Int32, n_qubits)
    end
    close(f)
    return Tuple(qubit_op_dict)
end

function read_ham_from_file(filename::String = "qubit_op.data")::Tuple{Vararg{Pair{Any, Any}}}
    #=
    Read the QubitOperator saved by QCQC.
    =#
    op = read_binary_qubit_op(filename)
    println("Qubit Hamiltonian read from $(filename)")

    counter = 0
    for (term, coeff) in op
        msg = abs(imag(coeff)) > 1e-14 && error("Only support real-valued Hamiltonian!")
        println("Term $(counter): ", term, " Coefficient: ", real(coeff))
        counter += 1
    end
    return op
end

struct MolecularHamiltonian <: Hamiltonian
    qubit_op::Tuple{Vararg{Pair{Any, Any}}}
    n_qubits::Int
end

MolecularHamiltonian(qubit_op::Tuple{Vararg{Pair{Any, Any}}}) = MolecularHamiltonian(qubit_op, count_qubits(qubit_op))

function _get_dense_matrix(h::MolecularHamiltonian, n_qubits::Int)
    #=
    Convert the qubit-Hamiltonian into a dense matrix.
    =#
    msg = n_qubits < h.n_qubits && error("h have more qubits than n_qubits!")
    qubit_op = h.qubit_op
    x_mat = [0.0im 1.0; 1.0 0.0]
    y_mat = [0.0im -1.0im; 1.0im 0.0]
    z_mat = [1.0 0.0im; 0.0 -1.0]
    i_mat = [1.0 0.0im; 0.0 1.0]
    symbol_to_mat = Dict(
        "X" => x_mat,
        "Y" => y_mat,
        "Z" => z_mat
    )
    mat = zeros(Float64, 2^n_qubits, 2^n_qubits)
    for (term, coeff) in qubit_op
        mat_list_tmp = [i_mat for i in 1:n_qubits]
        for (pos_, pauli) in term
            # Since the index from python starts from 0.
            pos = pos_ + 1
            mat_list_tmp[pos] = symbol_to_mat[pauli]
        end
        mat_tmp = reduce(LinearAlgebra.kron, mat_list_tmp) * coeff
        msg = LinearAlgebra.norm(imag(mat_tmp)) > 1e-14 && error("Complex values!")
        mat = mat + real(mat_tmp)
    end
    return mat
end

function _apply_term_on_basis(term::Tuple, coeff::ComplexF64, state::NNQS.ComputationBasis)
    res_coeff = coeff
    res_state = Base.copy(state)
    if length(term) == 0
        return res_state, res_coeff
    end
    res_dict = Dict(
        # (pauli, state) => (res_coeff, res_state)
        ("X", 1) => (1., -1),
        ("X", -1) => (1., 1),
        ("Y", 1) => (-1.0im, -1),
        ("Y", -1) => (1.0im, 1),
        ("Z", 1) => (-1., 1),
        ("Z", -1) => (1., -1),
    )
    for (pos_, pauli) in term
        # Since the index from python starts from 0.
        pos = pos_ + 1
        res_tmp = res_dict[pauli, state[pos]]
        res_coeff *= res_tmp[1]
        res_state[pos] = res_tmp[2]
    end
    msg = abs(imag(res_coeff)) > 1e-14 && error("Complex values!")
    return res_state, real(res_coeff)
end

function coupled_states(h::MolecularHamiltonian, state::NNQS.ComputationBasis)
    L = length(state)
    msg = L < h.n_qubits && error("The input state has too fewer qubits than the Hamiltonian!")
    qubit_op = h.qubit_op
    res_dict = Dict()
    for (term, coeff) in qubit_op
        new_state, new_coeff = _apply_term_on_basis(term, coeff, state)
        if new_state in keys(res_dict)
            res_dict[new_state] += new_coeff
        else
            res_dict[new_state] = new_coeff
        end
    end

    eps = 1e-14
    n_new_states = length(res_dict)
    c_states = zeros(Int, L, n_new_states)
    coefs = zeros(Float64, n_new_states)
    counter = 1
    for new_state in keys(res_dict)
        if abs(res_dict[new_state]) < eps
            continue
        end
        c_states[:, counter] = new_state
        coefs[counter] = res_dict[new_state]
        counter += 1
    end
    c_states = c_states[:, 1:counter - 1]
    coefs = coefs[1:counter - 1]
    return c_states, coefs
end

diagonal_coupling(h::MolecularHamiltonian, state::NNQS.ComputationBasis) = 0.


