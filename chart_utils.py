import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy.linalg
import itertools


def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x, dtype='float64')
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z


def neurons_surface_chart(path):
    with open(path) as f:
        X, Y, PK = [], [], []
        set_y, size = set(), 0

        for line in f:
            x, y, z = [int(s) for s in line.split(';')]
            set_y.add(y)
            X.append(x)
            Y.append(y)
            PK.append(z)
        size = len(set_y)

        X = np.array(X)
        Y = np.array(Y)
        PK = np.array(PK).reshape((size,-1))

        S1 = X.reshape((size, -1))
        S2 = Y.reshape((size, -1))

    Z = polyval2d(S1, S2, polyfit2d(S1.flatten(), S2.flatten(), PK.flatten()))

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('%PK')
    ax.set_zlim(0, 100)
    ax.set_xticks(S1[0])
    ax.set_yticks(S2[:,0])
    ax.plot_surface(S1, S2, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.scatter(X, Y, PK, c='r', s=50)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('%PK')
    ax.set_zlim(0, 100)
    ax.set_xticks(S1[0])
    ax.set_yticks(S2[:,0])
    ax.plot_surface(S1, S2, PK, rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    # ax.scatter(X, Y, PK, c='r', s=50)

    plt.show()