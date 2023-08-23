import pyscf
import numpy
import opt_einsum
import json


def get_mo_integrals(geometry: list, basis: str = "sto3g"):

    mol = pyscf.gto.M(
        atom=geometry, basis=basis,
        verbose=0, unit="angstrom")

    # Constant term.
    e0 = mol.energy_nuc()

    # Integrals in atomic orbital basis.
    one_body_ao = mol.intor("int1e_nuc") + mol.intor("int1e_kin")
    two_body_ao = mol.intor("int2e")

    # Perform Hartree-Fock to get molecular orbital coefficients. O(N^4).
    mf = pyscf.scf.RHF(mol).run()
    mo_coeff = mf.mo_coeff
    # mo_coeff can multiply with an arbitary unitary matrix u if needed (orbital localization):
    # mo_coeff = mo_coeff.dot(u)

    # Get integrals in molecular orbital basis.
    one_body_mo = opt_einsum.contract(
        one_body_ao, [0, 1],
        mo_coeff.conj(), [0, 10],
        mo_coeff, [1, 11],
        [10, 11]
    )
    two_body_mo = opt_einsum.contract(
        two_body_ao, [0, 1, 2, 3],
        mo_coeff.conj(), [0, 10],
        mo_coeff, [1, 11],
        mo_coeff.conj(), [2, 12],
        mo_coeff, [3, 13],
        [10, 11, 12, 13]
    )
    two_body_mo = numpy.moveaxis(
        two_body_mo,
        [0, 2, 3, 1],
        [0, 1, 2, 3]
    )

    # n_orb = mol.nao_nr()
    t = one_body_mo  # shape = [n_orb, n_orb]
    v = two_body_mo  # shape = [n_orb, n_orb, n_orb, n_orb]

    return e0, t, v


def save_to_binary(arr: numpy.ndarray, filename: str):
    # Save to a binary file.
    f = open(filename, "wb")
    size = f.write(arr.tobytes())
    f.close()
    return size


if __name__ == "__main__":
    # Unit in Angstrom.
    geometry_lih = ["H 0.0 0.0 0.0", "Li 0.0 0.0 1.0"]

    # geometry_lih = ["H 0.0 0.0 0.0", "Be 0.0 0.0 1.0", "H 0.0 0.0 2.0"]

    # geometry_co2 = [["O", 0.0, 0.0, -1.16], ["C", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.16]]

    # geometry_benzene_paper = [
    # ["C",  [+0.000000, +1.396792, 0.000000]],
    # ["C",  [+0.000000, -1.396792, 0.000000]],
    # ["C",  [+1.209657, +0.698396, 0.000000]],
    # ["C",  [-1.209657, -0.698396, 0.000000]],
    # ["C",  [-1.209657, +0.698396, 0.000000]],
    # ["C",  [+1.209657, -0.698396, 0.000000]],
    # ["H",  [+0.000000, +2.484212, 0.000000]],
    # ["H",  [+2.151390, +1.242106, 0.000000]],
    # ["H",  [-2.151390, -1.242106, 0.000000]],
    # ["H",  [-2.151390, +1.242106, 0.000000]],
    # ["H",  [+2.151390, -1.242106, 0.000000]],
    # ["H",  [+0.000000, -2.484212, 0.000000]],
    # ]


    e0, t, v = get_mo_integrals(geometry_lih, basis="sto3g")

    print(e0)
    # print(v.shape)

    path_name = 'data/lih.json'
    results = {'L':t.shape[0], 't':t.flatten(order='F').tolist(), 'v':v.flatten(order='F').tolist(), 'E0':e0}

    with open(path_name, 'w') as f:
        json.dump(results, f)

    # filename_t = "t.bin"
    # filename_v = "v.bin"
    # save_to_binary(t, filename_t)
    # save_to_binary(v, filename_v)
    # t_read = numpy.fromfile(filename_t, dtype=numpy.float64)
    # v_read = numpy.fromfile(filename_v, dtype=numpy.float64)
    # assert (numpy.isclose(
    #     numpy.linalg.norm(t.reshape(-1) - t_read.reshape(-1)), 0.0))
    # assert (numpy.isclose(
    #     numpy.linalg.norm(v.reshape(-1) - v_read.reshape(-1)), 0.0))
