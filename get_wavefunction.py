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
    Computes the wave function for a specific particle distribution. Only works if eta = 1 and q = 2.
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
    """
    Computes the wave function efficiently by vectorizing the computation.
    This method is much faster than the naive implementation "get_wavefunction_naive".
    Though admittedly this method is much less transparent and to the reader it might be very unclear whether
    the method correctly calculated the wavefunction. However, comparison with the naive implementation shows
    that they return the same wave function and hence that this method works correctly.
    :param z: List with lattice positions
    :param n_idx: List of particle positions
    :param q: System parameter
    :return:
    """
    import numpy as np
    N = len(z) # Number of sites
    n_particles = len(n_idx) # Number of particles
    eta = n_particles / N * q # Computes eta
    z = np.reshape(z, (1, N)) # Reshape z into a column vector
    z_particles = z[0, [n_idx]] # lattice positions with particles
    A = (z_particles -  np.transpose(z_particles)) ** q # Matrix used for computing the wave function in a vectorized manner
    B = z_particles - np.transpose(z) # Another matrix for computing the wave function via vectorization
    idx1, idx2 = np.where(B == 0) # Identify what entries of B is zero
    B[idx1, idx2] = 1 # Set these entries to 1 instead
    psi = np.prod(np.triu(A, k = 1) + np.tril(np.ones(np.shape(A)))) * np.prod(B ** (-eta)) # Compute the wave function
    return psi


if __name__ == '__main__':
    pass