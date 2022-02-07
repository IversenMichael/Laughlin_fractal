def get_wavefunction_naive(fill, q, z, n_idx):
    import numpy as np
    eta = fill * q
    psi = 1
    for i in n_idx:
        for j in n_idx:
            if i < j:
                psi *= (z[i] - z[j]) ** q
    for i in n_idx:
        psi *= np.prod((z[i] - z[:i]) ** (-eta)) * np.prod((z[i] - z[i+1:]) ** (-eta))
    return psi

def get_wavefunction(fill, q, z, n_idx):
    import numpy as np
    N = len(z)
    eta = fill * q
    z = np.reshape(z, (1, N))
    z_particles = z[0, [n_idx]]
    A = (z_particles -  np.transpose(z_particles)) ** q
    B = z_particles - np.transpose(z)
    idx1, idx2 = np.where(B == 0)
    B[idx1, idx2] = 1
    psi = np.prod(np.triu(A, k = 1) + np.tril(np.ones(np.shape(A)))) * np.prod(B ** (-eta))
    return psi


if __name__ == '__main__':
    import numpy as np
    from get_lattice import get_plane
    from time import time
    fill = 1 / 2
    q = 2
    z = get_fractal() / 4.33
    N = len(z)
    n_idx = np.sort(np.random.choice(list(range(N)), size=N//2, replace=False))
    print(get_wavefunction_naive(fill, q, z, n_idx))
    print(get_wavefunction(fill, q, z, n_idx))