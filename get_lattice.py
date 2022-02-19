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

def get_plane(side_length):
    """
    Computes the lattice points of a 9x9 grid represented by complex numbers
    :param side_length: Side length of grid, i.e. the grid contains side_length^2 sites
    :return:
    """
    import numpy as np
    total_size = side_length ** 2
    low = -(side_length - 1) / 2
    high = (side_length - 1) / 2 + 1
    z = (np.arange(low, high, 1).reshape((1, side_length)) + 1j*np.arange(low, high, 1).reshape((side_length, 1))).reshape((total_size, ))
    return z

def test_lattice():
    from matplotlib import pyplot as plt
    import numpy as np

    # ----------- Plane ----------- #
    side_length = np.random.randint(2, 20)
    z = get_plane(side_length)
    fig, ax = plt.subplots()
    ax.set_title(f'Plane with randomly chosen side length \n (side length = {side_length})')
    ax.grid()
    ax.axis('equal')
    ax.plot(np.real(z), np.imag(z), 'bo')

    # ----------- Fractal ----------- #
    z = get_fractal()
    fig, ax = plt.subplots()
    ax.set_title('Sierpinski carpet with 64 sites')
    ax.grid()
    ax.axis('equal')
    ax.plot(np.real(z), np.imag(z), 'bo')

if __name__ == '__main__':
    test_lattice()

