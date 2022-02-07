def get_fractal():
    import numpy as np
    offset_values = [-1, 0, 1]
    z_single = np.array([x + 1j * y for y in offset_values for x in offset_values if not (x == 0 and y == 0)])
    z = np.array([])
    for y in offset_values:
        for x in offset_values:
            if not (x == 0 and y == 0):
                z = np.concatenate([z, z_single + 3 * x + 3j * y])
    return z

def get_plane():
    import numpy as np
    z = (np.arange(-4, 5).reshape((1, 9)) + 1j*np.arange(-4, 5).reshape((9, 1))).reshape((9 * 9, ))
    return z

def reorder(z):
    z = 1j * np.sort(1j*z)
    z = - np.real(z) + 1j * np.imag(z)
    return z

if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    z = reorder(get_fractal())

    fig, ax  = plt.subplots()
    ax.grid()
    ax.plot(np.real(z), np.imag(z), 'bo', markersize=15)
    ax.axis('equal')
    fig.show()

