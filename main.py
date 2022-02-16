def main():
    import numpy as np
    import sys
    sys.path.append('C:\\Users\\au544901\\Documents\\GitHub\\Laughlin_fractal')
    from get_lattice import get_plane
    q = 2
    N = 8
    n_particles = 4
    if 'psi' in locals():
        print('psi was not recomputed')
    else:
        print('Computing psi ...')
        psi, basis, basis_dict = get_psi(N, n_particles, q, z)
        print('psi has been computed')

    operators = []

    active_sites = list(range(N))
    interaction_sites = [[i, j] for i in range(N) for j in range(N) if i != j]
    operator_dict = dict()

    dict_counter = 0
    print('Calculating number operators')
    for site in active_sites:
        print(f'site = {site}')
        operators.append(number_operator(basis, site).tocsc())
        operator_dict[dict_counter] = f'Potential {site}'
        dict_counter += 1

    print('Calculating interaction operators')
    for site1, site2 in interaction_sites:
        print(f'site1 = {site1}, site2 = {site2}')
        operators.append(interaction(basis, site1, site2).tocsc())
        operator_dict[dict_counter] = f'Interaction {site1}, {site2}'
        dict_counter += 1

    print('Calculating hopping operators')
    for site1, site2 in interaction_sites:
        print(f'site1 = {site1}, site2 = {site2}')
        operators.append(hopping(basis, basis_dict, site1, site2).tocsc())
        operator_dict[dict_counter] = f'Hopping {site1}, {site2}'
        dict_counter += 1

    psi_T = np.conj(np.transpose(psi))
    C = np.zeros((len(operators), len(operators)), dtype=np.complex128)
    for i, o1 in enumerate(operators):
        for j, o2 in enumerate(operators):
            #print(f'i = {i}, j = {j}')
            C[i, j] = (psi_T @ o1 @ o2 @ psi) - (psi_T @ o1 @ psi) * (psi_T @ o2 @ psi)

    D, P = np.linalg.eig(C)
    idx = np.where(np.abs(D) < 1e-12)[0]
    V = P[:, idx]
    V_sparse = get_sparse_vectors(V)
    # np.savetxt('C', C)

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
    from scipy.sparse import lil_matrix
    n = lil_matrix((len(basis), len(basis)))
    for i, b in enumerate(basis):
        if site in b:
            n[i, i] = 1
    return n


def interaction(basis, site1, site2):
    from scipy.sparse import lil_matrix
    inter = lil_matrix((len(basis), len(basis)))
    for i, b in enumerate(basis):
        if site1 in b and site2 in b:
            inter[i, i] = 1
    return inter


def hopping(basis, basis_dict, site1, site2):
    from scipy.sparse import lil_matrix
    hop = lil_matrix((len(basis), len(basis)))
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


def get_sparse_vectors(V):
    """
    Consider a subspace spanned by the vectors down each column of V, i.e. V[:, [i]]. This methods determines a collection of sparse vectors in this subspace
    The algorithm was found in the paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7547961&tag=1.

    :param V: numpy.ndarray containing vectors down each column
    :return:  numpy.ndarray with sparse vectors down each column.
    """
    import numpy as np
    # ----------------- HELPER FUNCTIONS ----------------- #
    def adm(Y, q_init, l, MaxIter, tol):
        import numpy as np
        q = q_init
        for k in range(MaxIter):
            q_old = q
            x = soft_thresholding(Y @ q, l)
            q = np.transpose(Y) @ x / norm(np.transpose(Y) @ x)
            res_q = norm(q_old - q)
            if res_q <= tol:
                return q
        print('No convergence')

    def soft_thresholding(X, d):
        import numpy as np
        return np.sign(X) * np.maximum(np.abs(X) - d, 0)

    # ----------------- ALGORITHM ----------------- #
    p, n = np.shape(V)
    l = 1 / np.sqrt(p)
    MaxIter = 10000
    tol_adm = 1e-5
    q_mtx = np.zeros((n, p), dtype=np.complex128)
    for i in range(p):
        q_0 = np.transpose( V[[i], :]/ norm(V[[i], :]) )
        q_mtx[:, [i]] = adm(V, q_0, l , MaxIter, tol_adm)

    # ----------------- REMOVE DUPLICATES ----------------- #
    for i in range(p):
        idx = np.where(q_mtx[:, i]!=0)[0]
        q_mtx[:, [i]] = np.sign(q_mtx[idx[0], i])*q_mtx[:, [i]]
    q_mtx = np.unique(np.round(q_mtx, 5), axis=1)

    sparse_vectors = np.round(V@q_mtx, 5)
    return sparse_vectors

def norm(v):
    """
    Returns the norm/length of a vector. If v is the zero vector, the method returns 1.
    :param v: np.ndarray of size v.shape = (N, 1)
    :return: norm of vector
    """
    import numpy as np
    n = np.sqrt(np.sum(v ** 2)) # Compute norm
    if n == 0:
        return 1                # If v is zero vector, return 1.
    else:
        return n                # If v is not zero vector, return its norm.


if __name__ == '__main__':
    pass