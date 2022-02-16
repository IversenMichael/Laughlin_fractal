def get_wavefunction_naive(z, n_idx, q):
    """
    Computes the wave function for a specific particle distribution
    :param z: Position of sites
    :param n_idx: List of particle positions (occupied sites)
    :return:
    """
    import numpy as np
    filling = len(n_idx)/len(z)
    eta = filling * q
    psi = np.array([1], dtype=np.complex128) # Initialize psi
    for i in n_idx: # Loop though all particles
        for j in n_idx: # --||--
            if i < j:
                psi *= (z[i] - z[j]) ** q # Compute the first factor
    for i in n_idx: # Loop through all particles
        for j in range(len(z)):
            if i != j:
                psi *= (z[i] - z[j]) ** (-eta) # Compute second factor
    return psi

def get_wavefunction_naive_restricted(z, n_idx, q):
    """
    Computes the wave function for a specific particle distribution. Only works if eta = 1.
    :param z: Position of sites
    :param n_idx: List of particle positions (occupied sites)
    :return:
    """
    import numpy as np
    filling = len(n_idx)/len(z)
    eta = filling * q
    psi = np.array([1], dtype=np.complex128) # Initialize psi
    for i in n_idx: # Loop though all particles
        for j in n_idx: # --||--
            if i < j:
                psi *= (z[i] - z[j]) ** 2 # Compute the first factor
    for i in n_idx: # Loop through all particles
        psi *= z[i] # Compute second factor
    return psi

def get_wavefunction(z, n_idx, q):
    import numpy as np
    N = len(z)
    n_particles = len(n_idx)
    eta = n_particles / N * q
    z = np.reshape(z, (1, N))
    z_particles = z[0, [n_idx]]
    A = (z_particles -  np.transpose(z_particles)) ** q
    B = z_particles - np.transpose(z)
    idx1, idx2 = np.where(B == 0)
    B[idx1, idx2] = 1
    psi = np.prod(np.triu(A, k = 1) + np.tril(np.ones(np.shape(A)))) * np.prod(B ** (-eta))
    return psi


if __name__ == '__main__':
    pass