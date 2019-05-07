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


def surface_chart(data_path, label_x, label_y, label_z):
    with open(data_path) as f:
        X, Y, data_3 = [], [], []
        set_y, size = set(), 0

        for line in f:
            x, y, z = [int(s) for s in line.split(';')]
            set_y.add(y)
            X.append(x)
            Y.append(y)
            data_3.append(z)
        size = len(set_y)

        X = np.array(X)
        Y = np.array(Y)
        data_3 = np.array(data_3).reshape((size,-1))

        data_1 = X.reshape((size, -1))
        data_2 = Y.reshape((size, -1))

    Z = polyval2d(data_1, data_2, polyfit2d(data_1.flatten(), data_2.flatten(), data_3.flatten()))

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z, rotation='vertical')
    ax.set_zlim(0, 100)
    ax.set_xticks(data_1[0])
    ax.set_yticks(data_2[:,0])
    ax.plot_surface(data_1, data_2, data_3, rstride=1, cstride=1, cmap='inferno', edgecolor='none')
    ax.scatter(X, Y, data_3, c='r', s=50, alpha=0.5)

    plt.show()