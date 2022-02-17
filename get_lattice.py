def get_fractal():
    """
    Computes the lattice positions represented by complex numbers of the sierpinski carpet with 64 lattice points.
    :return:
    """
    import numpy as np
    offset_values = [-1, 0, 1]
    z_single = np.array([x + 1j * y for y in offset_values for x in offset_values if not (x == 0 and y == 0)]) # Draws a single square of lattice points
    z = np.array([]) # Container for lattice positions
    for y in offset_values: # Loops over all offsets
        for x in offset_values: # --||--
            if not (x == 0 and y == 0): # We dont want the middle filled out
                z = np.concatenate([z, z_single + 3 * x + 3j * y]) # Add next lattice point
    return z

def get_plane():
    """
    Computes the lattice points of a 9x9 grid represented by complex numbers
    :return:
    """
    import numpy as np
    z = (np.arange(-4, 5).reshape((1, 9)) + 1j*np.arange(-4, 5).reshape((9, 1))).reshape((9 * 9, ))
    return z

if __name__ == '__main__':
    pass

