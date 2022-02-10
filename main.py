def main():
    import numpy as np
    from get_lattice import get_plane
    n_particles = 4
    q = 2
    z = get_plane() / 4.33
    N = len(z)
    if 'psi' in locals():
        print('psi was not recomputed')
    else:
        print('Computing psi ...')
        psi, basis, basis_dict = get_psi(N, n_particles, q, z)
        print('psi has been computed')

    operators = []

    print('Calculating number operators')
    for site in [1, 8, 9, 10, 17]:
        print(f'site = {site}')
        operators.append(number_operator(basis, site))

    print('Calculating interaction operators')
    for site1 in [1, 8, 9, 10, 17]:
        for site2 in [1, 8, 9, 10, 17]:
            if site1 != site2:
                print(f'site1 = {site1}, site2 = {site2}')
                operators.append(interaction(basis, site1, site2))
    print('Calculating hopping operators')
    for site1 in [1, 8, 9, 10, 17]:
        for site2 in [1, 8, 9, 10, 17]:
            if site1 != site2:
                print(f'site1 = {site1}, site2 = {site2}')
                operators.append(hopping(basis, site1, site2))

    psi_T = np.conj(np.transpose(psi))
    C = np.zeros((len(operators), len(operators)), dtype=np.complex128)
    for i, o1 in enumerate(operators):
        for j, o2 in enumerate(operators):
            print(f'i = {i}, j = {j}')
            C[i, j] = (psi_T @ o1 @ o2 @ psi) - (psi_T @ o1 @ psi) * (psi_T @ o2 @ psi)

    np.savetxt('C', C)


def get_psi(N, n_particles, q, z):
    import numpy as np
    from get_wavefunction import get_wavefunction
    from itertools import combinations
    basis = list(combinations(range(N), n_particles))
    basis_dict = dict()
    for i, b in enumerate(basis):
        basis_dict[b] = i

    psi = np.zeros(len(basis), dtype=np.complex128)
    for i, b in enumerate(basis):
        if i%1e5 == 0:
            print(f'{i/(1.6*1e6)*100}%')
        psi[i] = get_wavefunction(n_particles, q, z, b)
    del i, b
    norm = np.sqrt(np.sum(np.conj(np.transpose(psi)) @ psi))
    psi /= norm
    return psi, basis, basis_dict


def number_operator(basis, site):
    from scipy.sparse import lil_array
    n = lil_array((len(basis), len(basis)))
    for i, b in enumerate(basis):
        if site in b:
            n[i, i] = 1
    return n


def interaction(basis, site1, site2):
    from scipy.sparse import lil_array
    inter = lil_array((len(basis), len(basis)))
    for i, b in enumerate(basis):
        if site1 in b and site2 in b:
            inter[i, i] = 1
    return inter


def hopping(basis, basis_dict, site1, site2):
    from scipy.sparse import lil_array
    hop = lil_array((len(basis), len(basis)))
    for i, b in enumerate(basis):
        if site1 in b and site2 not in b:
            b_new = tuple(sorted(b[:b.index(site1)] + b[b.index(site1) + 1:] + (site2, )))
            j = basis_dict[b_new]
            hop[j, i] = 1
        if site1 not in b and site2 in b:
            b_new = tuple(sorted(b[:b.index(site2)] + b[b.index(site2) + 1:] + (site1, )))
            j = basis_dict[b_new]
            hop[j, i] = 1
    return hop


if __name__ == '__main__':
    main()