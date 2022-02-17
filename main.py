def main():
    import numpy as np
    import sys
    sys.path.append('C:\\Users\\au544901\\Documents\\GitHub\\Laughlin_fractal')
    from get_lattice import get_fractal
    from get_psi import get_psi

    # ---------------------- Initialize system ---------------------- #
    z = get_fractal()
    N = len(z)
    n_particles = 4
    q = 2
    psi, basis, basis_dict = get_psi(N, n_particles, q, z)
    z_dict = dict()
    for i, x in enumerate(z):
        z_dict[x] = i

    # ---------------------- Calculate operators ---------------------- #
    generator = operator_generator(basis, z)
    sites = []
    D = []
    P = []
    operators = next(generator)

    # ------------ Calculate the correct linear combination of operators ------------ #
    print(f"site1 = {site1}, site2 = {site2}")
    d, p = find_combination(psi, operators)
    o = sum([p[i, -1]*operators[i] for i in range(len(p))])
    psi_T = np.transpose(np.conj(psi))
    print(psi_T@o@o@psi - (psi_T@o@psi)**2)

def operator_generator(basis, z):
    """
    Generates a list of operators
    :param basis: List of basis elements
    :param z: List of lattice positions
    :return:
    """
    operators = [] # Container for operators
    for site1, z1 in enumerate(z): # Loop through all lattice sites
        operators = operators + [number_operator(basis, site1)] # Append the number operator on site 1
        print(f"site1 = {site1}")
        for offset in [1, 1j]: # Site 1 can interact with the site above it and to the right of it.
            z2 = z1 + offset # Find the position of a neighbouring site
            if z2 in z: # If this site is a part of the lattice
                site2 = z_dict[z2] # Find the index of this lattice site
                # Append the interaction and hopping between these sites
                operators = operators + [interaction(basis, site1, site2), hopping(basis, basis_dict, site2, site1) + hopping(basis, basis_dict, site1, site2)]
    yield operators

def find_combination(psi, operators):
    """
    Uses the method from reference https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031029 to find
    the linear combinations of operators which has psi as an eigenvector
    :param psi: Wave function
    :param operators: List of operators
    :return:
    """
    import numpy as np
    psi_T = np.conj(np.transpose(psi)) # Compute the hermitian conjugate of psi
    C = np.zeros((len(operators), len(operators)), dtype=np.complex128) # Initialize correlation matrix
    for i, o1 in enumerate(operators): # Loop through all operators
        print(f"i = {i}")
        for j, o2 in enumerate(operators): # --||--
            C[i, j] = (psi_T @ o1 @ o2 @ psi) - (psi_T @ o1 @ psi) * (psi_T @ o2 @ psi) # Compute the current matrix element

    D, P = np.linalg.eig(C) # Diagonalize C to find the zero eigenvalues
    return D, P # Return eigenvalues and eigenvectors.

def number_operator(basis, site):
    """
    Computes the number operator on a site in a basis
    These matrices are very large, so we use sparse matrices.
    Note that lil_matrix efficiently builds a matrix while csc_matrix efficiently does calculations.
    :param basis: List of basis elements
    :param site: The site we compute the number operator for.
    :return:
    """
    from scipy.sparse import lil_matrix
    n = lil_matrix((len(basis), len(basis))) # Initialization of operator
    for i, b in enumerate(basis): # Loop through all basis elements
        if site in b: # If the site is in the basis
            n[i, i] = 1 # Add 1
    return n.tocsc() # Convert to csc format so arithmetic is efficient

def interaction(basis, site1, site2):
    """
    Computes the interaction operator between site 1 and site 2
    :param basis: List of basis elements
    :param site1: Site 1
    :param site2: Site 2
    :return:
    """
    from scipy.sparse import lil_matrix
    inter = lil_matrix((len(basis), len(basis))) # Initialize the interaction operator
    for i, b in enumerate(basis): # Loop though all basis elemnts
        if site1 in b and site2 in b: # If both site 1 and site 2 are in the basis element they interact
            inter[i, i] = 1 # Add 1
    return inter.tocsc()

def hopping(basis, basis_dict, site1, site2):
    """
    Computes the hopping operator between site 1 and site 2.
    Note that this operator is NOT hermitian. Add the conjugate to make it hermittian
    ie. hopping(basis, basis_dict, site1, site2) + hopping(basis, basis_dict, site2, site1) IS hermittian.
    :param basis: List of basis elements
    :param basis_dict: Dictionary which returns the index in the reduced basis corresponding to a basis element
    :param site1: Site 1
    :param site2: Site 2
    :return:
    """
    from scipy.sparse import lil_matrix
    hop = lil_matrix((len(basis), len(basis))) # Initialization of hopping operator
    for i, b in enumerate(basis): # Loop through all basis elements
        if site1 in b and site2 not in b: # If site 1 is in the basis element but site 2 is not, then the particle on site 1 can hop to site 2
            b_new = tuple(sorted(b[:b.index(site1)] + b[b.index(site1) + 1:] + (site2, ))) # Remove site 1 from the basis element, add site 2.
            j = basis_dict[b_new] # Find the index of this new basis element
            hop[j, i] = 1 # Add 1 to this entry in the matrix.
    return hop.tocsc()

if __name__ == '__main__':
    pass