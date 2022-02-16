def main():
    import numpy as np
    N = 100
    l = 7
    z = np.array([np.exp(1j*2*np.pi/N * l) - np.exp(1j*2*np.pi/N * j) for j in range(N) if j != l])
    print(1/np.prod(z))
    print(np.exp(1j*2*np.pi/N * l))

if __name__ == '__main__':
    main()