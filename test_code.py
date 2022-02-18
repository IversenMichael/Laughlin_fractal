def main():
    """
    This function tests the methods from references 
    - https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031029
    - https://journals.aps.org/prb/pdf/10.1103/PhysRevB.98.081113
    for the first example in the second reference (the Haldane-Shastry model).
    We also test if the three functions in get_wavefunction computes the same Haldane-Shastry wavefunction.    
    :return: 
    """
    import numpy as np
    import sys
    np.set_printoptions(precision=4)
    from get_wavefunction import get_wavefunction_naive, get_wavefunction_naive_restricted, get_wavefunction
    from get_psi import get_psi
    # ------------ Initialize quantities ------------ #
    N = 10 # System size
    n_particles = N//2 # Number of particles (spin up) in system
    q = 2 # System parameter
    z = [np.exp(1j * 2 * np.pi / N * k) for k in range(N)] # Lattice sites
    psi, basis, basis_dict = get_psi(N, n_particles, q, z) # Wave function at these lattice sites.

    operators = [] # Container for operators
    operator_dict = dict() # Operator dictionary for looking up what the operators actually are
    for dict_counter, sep in enumerate(range(0, N//2 + 1)): # Loop through all separations.
        print(f'sep = {sep}') # Print separation of current interation
        operators.append(get_H(sep, N, basis, basis_dict).tocsc()) # Compute the operator corresponding to this separation
        operator_dict[dict_counter] = f'sep={sep}' # Write an explanatory text and add to the operator dictionary.

    operators[-1] /= 2 # The last operator actually contains all terms twice. So I divide by 2.

    print('')
    print('# ---------------- First method ---------------- #')
    # First we test the method from the first article
    psi_T = np.conj(np.transpose(psi)) # Hermitian conjugate of psi
    C = np.zeros((len(operators), len(operators)), dtype=np.complex128) # Container for C matrix
    for i, o1 in enumerate(operators): # Loop through all operators
        for j, o2 in enumerate(operators): # Loop through all operators
            C[i, j] = (psi_T @ o1 @ o2 @ psi) - (psi_T @ o1 @ psi) * (psi_T @ o2 @ psi) # Compute the correlation

    D, P = np.linalg.eig(C) # Diagonalize
    if np.sum(np.abs(np.imag(D))) >1e-12 or np.sum(np.abs(np.imag(P))) >1e-12:
        # Test if D and P are real
        print('Significant imaginary value')
    else:
        D = np.real(D)
        P = np.real(P)
    print(f'D = {D}')
    idx = np.where(np.abs(D) < 1e-12)[0] # Find the zero eigenvalues
    print(f'idx = {idx}')
    V1 = P[:, idx] # Extract the eigenvectors corresponding to zero eigenvalues
    print(f'V = \n{V1}')

    print('')
    print('# ---------------- Second method ---------------- #')
    psi_T = np.conj(np.transpose(psi)) # Hermitian conjugate of psi
    # Reuse the operators from last method but we have to append the identity.
    # Note that we are doing somethigh slightly different from the article. They have the identity as the first operator. We have it as the last.
    operators.append(np.identity(len(psi), dtype=np.complex128))
    M = np.zeros((len(operators), len(operators)), dtype=np.complex128) # Container for the M matrix
    for i, o1 in enumerate(operators): # Loop through all operators
        for j, o2 in enumerate(operators): # Loop through all operators
            M[i, j] = psi_T @ np.transpose(np.conj(o1)) @ o2 @ psi # Compute the entries of the M matrix

    D, P = np.linalg.eig(M) # Diagonalize
    print(f'D = {D}')
    idx = np.where(np.abs(D)<1e-12)[0] # Find the zero eigenvalues
    print(f'idx = {idx}')
    V2 = P[:, idx] # Extract the eigenvectors corresponding to zero eigenvalues.
    print(f'V = \n{V2}')

    psi = psi.reshape((len(psi), 1))

    print('')
    print('# ---------- Is |psi> an eigenstate of Ha and Hb? ----------  #')
    Ha = get_Ha(N, basis, basis_dict)
    Ha_psi = Ha @ psi
    norm = np.sqrt(np.conj(np.transpose(Ha_psi)) @ Ha_psi)[0, 0]
    Ha_psi /= norm

    print(f'Is |psi> an eigenstate of Ha? \t Answer = {np.sum(np.abs(Ha_psi) - np.abs(psi)) < 1e-9}')

    Hb = get_Hb(N, basis, basis_dict)
    print(f'Does Hb annihilate |psi>? \t\t Answer = {np.sum(np.abs(Hb @ psi)) < 1e-9}')

    print('')
    print('# ---------- Does Ha hide within V? ----------  #')
    v = np.ones((N//2 + 1, 1))
    v /= np.sqrt(np.transpose(v) @ v)
    Q1, R = np.linalg.qr(V1)
    boolian = np.sum(np.abs(sum([(np.transpose(v) @ Q1[:, [i]]) * Q1[:, [i]] for i in range(Q1.shape[1])]) - v)) < 1e-9
    print(f'Does Ha hide within V1? \t\t Answer = {boolian}')
    Q2, R = np.linalg.qr(V2[:-1, :])
    boolian = np.sum(np.abs(sum([(np.transpose(v) @ Q2[:, [i]]) * Q2[:, [i]] for i in range(Q2.shape[1])]) - v)) < 1e-9
    print(f'Does Ha hide within V2? \t\t Answer = {boolian}')

    print('')
    print('# ---------- Does Hb hide within V? ----------  #')
    E0 = - np.pi**2 / 24 * (N + 5 / N)
    prefactor = (2 * np.pi / N) ** 2
    chord_distance = [4 * np.sin(np.pi * n / N) ** 2 for n in range(1, N // 2 + 1)]
    v = np.zeros((N//2 + 1, 1))
    for i, d in enumerate(chord_distance):
        v[i+1] = prefactor/d
    boolian = np.sum(np.abs(sum([(np.transpose(v) @ Q1[:, [i]]) * Q1[:, [i]] for i in range(Q1.shape[1])]) - v)) < 1e-9
    print(f'Does Hb hide within V1? \t\t Answer = {boolian}')
    boolian = np.sum(np.abs(sum([(np.transpose(v) @ Q2[:, [i]]) * Q2[:, [i]] for i in range(Q2.shape[1])]) - v)) < 1e-9
    print(f'Does Hb hide within V2? \t\t Answer = {boolian}')

    print('')
    print('# ---------- Is the wave function computed correctly? ----------  #')
    q = 2
    n_tests = 100
    error = 0
    for _ in range(n_tests):
        b = np.random.choice(range(N), N//2, replace=False)
        psi_naive = get_wavefunction_naive(z, b, q)
        psi = get_wavefunction(z, b, q)
        error += np.abs(psi - psi_naive)**2
    boolian = error[0] < 1e-9
    print(f"Does different methods (2x naive + vectorized) yield same result? \t\t Answer = {boolian}")

def get_H(sep, N, basis, basis_dict):
    """
    Computes operators H_i given separation
    :param sep: Separation
    :param N: Number of sites
    :param basis:
    :param basis_dict: Dictionary for converting basis states to index in the reduced basis (with fixed number of particles)
    :return:
    """
    from scipy.sparse import lil_matrix # We use sparse matrices
    H = lil_matrix((len(basis), len(basis))) # Initialize operator
    for i, b in enumerate(basis): # Loop through all basis elements. We will act with the operator on these states. i is the index of b in the reduced basis
        for site1 in range(N): # Loop through all sites in the system
            site2 = (site1 + sep) % N # site1 and site2 are separated by sep sites with periodic boundary condition.
            # ------------ S^z_site1 * S^z_site2 ------------ #
            if ((site1 in b) and (site2 in b)) or ((site1 not in b) and (site2 not in b)):
                H[i, i] += 1 / 4 # If site1 and site2 are both spin up or spin down the result is + 1/4
            else:
                H[i, i] += - 1 / 4 # If site1 and site2 contains one spin up and one spin down the result is - 1/4

                # --------------- S^x_site1 * S^x_site2  + S^y_site1 * S^y_site2) ---------------- # These two expression are equal
                # ------------ 1/2 * (S^+_site1 * S^-_site2  + S^-_site1 * S^+_site2) ------------ # But we use the last one
                # Note that we are already in the case where site1 and site2 contains exactly one spin up and one spin down. Hence these can flip.
                if (site1 in b) and (site2 not in b): # If site1 is spin up and site2 is spin down.
                    b_new = tuple(sorted(b[:b.index(site1)] + b[b.index(site1) + 1:] + (site2,))) # Remove site1 from b and add site2.
                    j = basis_dict[b_new] # find the index in the reduced basis of the new state
                    H[j, i] += 1 / 2 # Add 1/2 to the operator.
                else: # Otherwise site1 is spin down and site2 is spin up.
                    b_new = tuple(sorted(b[:b.index(site2)] + b[b.index(site2) + 1:] + (site1,))) # In that case we remove site2 from b and add site1
                    j = basis_dict[b_new] # Find index of b_new in the reduced basis.
                    H[j, i] += 1 / 2 # Add 1/2 to the operator.
    return H

def get_Ha(N, basis, basis_dict):
    from scipy.sparse import lil_matrix
    Ha = lil_matrix((len(basis), len(basis)))
    for sep in range(N//2):
        Ha += get_H(sep, N, basis, basis_dict)
    Ha += 0.5 * get_H(N//2, N, basis, basis_dict)
    return Ha

def get_Hb(N, basis, basis_dict):
    import numpy as np
    from scipy.sparse import lil_matrix
    E0 = - np.pi**2 / 24 * (N + 5 / N)
    prefactor = (2 * np.pi / N) ** 2
    chord_distance = [4 * np.sin(np.pi * n / N) ** 2 for n in range(1, N // 2 + 1)]
    Hb = lil_matrix((len(basis), len(basis)), dtype=np.complex128)
    for d, sep in zip(chord_distance[:-1], range(1, N//2)):
        Hb += get_H(sep, N, basis, basis_dict) / d
    Hb += get_H(N//2, N, basis, basis_dict) / (2 * chord_distance[-1])
    Hb *= prefactor
    Hb += -E0 * np.identity(len(basis), dtype=np.complex128)
    return Hb

if __name__ == '__main__':
    main()