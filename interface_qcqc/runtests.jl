include("./mol_hamiltonian.jl")

using Test

function test_direct_diag()
    # To check if the Hamiltonian is correctly read in.
    op = read_binary_qubit_op("qubit_op.data")
    ham = MolecularHamiltonian(op)
    mat = _get_dense_matrix(ham, ham.n_qubits)
    eigs = LinearAlgebra.eigvals(mat)
    # Calculated by PySCF for H2 (R=2.0 Angstrom, STO-3G)
    e_fci = -0.9486411121761857
    return abs(eigs[1] - e_fci) <= 1e-10
end

function _check_state(mat::Matrix{Float64}, input_state::NNQS.ComputationBasis,
                      output_states::Matrix{Int}, output_coefs::Vector{Float64})
    ONE = reshape([0.0, 1.0], (2, 1))
    ZERO = reshape([1.0, 0.0], (2, 1))
    basis_dict = Dict(
        1 => ONE,
        -1 => ZERO,
    )
    vec_tmp = [basis_dict[i] for i in input_state]
    vec = reduce(LinearAlgebra.kron, vec_tmp)
    vec = mat * vec
    vec_check = zeros(Float64, size(vec)[1])
    n_output_states = length(output_coefs)
    for i in 1:n_output_states
        vec_tmp_i = [basis_dict[ii] for ii in output_states[:, i]]
        vec_i = reduce(LinearAlgebra.kron, vec_tmp_i)
        vec_i = vec_i .* output_coefs[i]
        vec_check = vec_check + vec_i
    end
    err = LinearAlgebra.norm(vec_check - vec)
    return err
end

function test_coupled_states()
    op = read_binary_qubit_op("qubit_op.data")
    ham = MolecularHamiltonian(op)
    mat = _get_dense_matrix(ham, ham.n_qubits)
    n_qubits = ham.n_qubits
    is_pass = true
    for i in 0:(2^n_qubits - 1)
        bitstring = string(i, base=2)
        bitstring = ('0'^(n_qubits - length(bitstring))) * bitstring
        state = [Int(i - '0') * 2 - 1 for i in bitstring]
        c_states, coefs = coupled_states(ham, state)
        err = _check_state(mat, state, c_states, coefs)
        is_pass = (err < 1e-10) && is_pass
    end
    return is_pass
end

@testset "Testing..." begin
    @test test_direct_diag()
    @test test_coupled_states()
end