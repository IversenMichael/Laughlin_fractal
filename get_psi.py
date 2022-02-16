def get_psi(N, n_particles, q, z):
    """
    This function computes the wave function for each basis element.
    :param N: Size of system
    :param n_particles: Number of particles in system
    :param q: System parameter
    :param z: list of site coordinates
    :return:
    """
    import numpy as np
    from get_wavefunction import get_wavefunction
    from itertools import combinations

    basis = list(combinations(range(N), n_particles)) # Compute all basis states with n_partices on N sites
    basis_dict = dict() # Initialize a dictionary for translating a given basis state into the index in the reduced basis
    for i, b in enumerate(basis):
        basis_dict[b] = i # Loop through all basis states and assign each basis state its index in the reduced basis.

    psi = np.zeros(len(basis), dtype=np.complex128) # Initialize the wave function
    for i, b in enumerate(basis):
        psi[i] = get_wavefunction(z, b, q) # Loop through all basis states and compute the wave function at this state. Save the result in psi[i].
    del i, b
    norm = np.sqrt(np.conj(np.transpose(psi)) @ psi) # Compute the norm of psi
    psi /= norm # Normalize psi
    return psi, basis, basis_dict