import sys
# QCQC: https://gitlab.com/auroraustc/qcqc.git
sys.path.append("./qcqc/")

import openfermion

import pyscf_helper
import utils


# Unit for length: Angstrom
# Unit for energy: Hartree
#geometry = [["H", [0.0, 0.0, 2.0 * i]] for i in range(2)]
dist = 2.0
geometry = [["O", [0., 0., 0.]], ["C", [0., 0., 1. * dist]], ["O", [0., 0., 2. * dist]]]
res = pyscf_helper.init_scf(
    geometry=geometry, run_fci=True, run_rccsd=True)
qubit_op = None
for i in res:
    if type(i) is openfermion.QubitOperator:
        qubit_op = i
        break

# Not used.
mol = res[0]
n_qubits = openfermion.count_qubits(qubit_op)
n_electrons = mol.nelectron
print("n_qubits: ", n_qubits, " n_electrons: ", n_electrons)
particle_num_op = utils.particle_number_operator(n_qubits)
particle_num_op = openfermion.jordan_wigner(particle_num_op)
total_spin_op = utils.total_spin_operator(n_qubits, n_electrons)
total_spin_op = openfermion.jordan_wigner(total_spin_op)

filename = "qubit_op.data"
utils.save_binary_qubit_op(qubit_op, filename=filename)
print("Qubit Hamiltonian saved to %s." % (filename))
utils.save_binary_qubit_op(particle_num_op, filename="qubit_op_n.data")
utils.save_binary_qubit_op(total_spin_op, filename="qubit_op_s.data")
