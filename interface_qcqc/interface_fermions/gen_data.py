import sys
sys.path.append("./qcqc")

import openfermion
import pyscf
import pyscf.fci
import numpy
import json

import pyscf_helper
import utils


# Define the molecule geometry and the basis set.
# Note that some molecules (e.g., C2) have spin = 2 for its ground state.
dist = 1.0
geometry = [
    ["H", [0., 0., 0.]],
    ["Li", [0., 0., dist]],
]
basis = "sto3g"
spin = 0

mol = pyscf.gto.M(atom=geometry, basis=basis, spin=spin)
mf = pyscf.scf.RHF(mol).run()
e_hf = mf.e_tot
print("Hartree-Fock energy: ", e_hf)

if mol.nao_nr() <= 16:
    mf_fci = pyscf.fci.FCI(mf).run()
    e_fci = mf_fci.e_tot
    print("FCI energy: ", e_fci)
else:
    print("Skip FCI calculation.")


e0 = mol.energy_nuc()
one_body_mo_, two_body_mo_ = pyscf_helper.get_mo_integrals_from_molecule_and_hf_orb(
    mol=mol, mo_coeff=mf.mo_coeff)

one_body_mo = one_body_mo_
two_body_mo = two_body_mo_

#------------------------------------------#
# This is read by Fermions to perform DMRG.#
#------------------------------------------#
t = one_body_mo
v = numpy.moveaxis(two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3])
path_name = "ham_coeff_for_fermions.json"
results = {
    "L":t.shape[0],
    "t":t.flatten(order="F").tolist(),
    "v":v.flatten(order="F").tolist(),
    "E0":e0,
}

with open(path_name, "w") as f:
    json.dump(results, f)
#==========================================#


#------------------------------------------#
# This is read by NNQS.                    #
#------------------------------------------#
ham_ferm_op_1, ham_ferm_op_2 = pyscf_helper.get_hamiltonian_ferm_op_from_mo_ints(
    one_body_mo=one_body_mo,
    two_body_mo=two_body_mo)
ham_ferm_op = ham_ferm_op_1 + ham_ferm_op_2
ham_ferm_op += e0
ham_qubit_op = openfermion.jordan_wigner(ham_ferm_op)
_tmp = utils.save_binary_qubit_op(op=ham_qubit_op, filename="ham_qubit_op_for_nnqs.data")
#==========================================#
