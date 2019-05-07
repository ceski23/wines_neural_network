import numpy as np

def initnw(layer_size, input_size):
    beta = 0.7 * (layer_size ** (1.0 / input_size))
    w_rand = np.random.rand(layer_size, input_size) * 2 - 1
    w_rand = np.sqrt(1.0 / np.square(w_rand).sum(axis=1).reshape(layer_size, 1)) * w_rand

    w = beta * w_rand
    b = beta * np.linspace(-1, 1, layer_size) * np.sign(w[:, 0])

    # amin, amax = -1, 1
    # x = 0.5 * (amax - amin)
    # y = 0.5 * (amax + amin)
    # w = x * w
    # b = x * b + y

    # minmax = np.full((input_size, 2), np.array([-1, 1]))
    # x = 2. / (minmax[:, 1] - minmax[:, 0])
    # y = 1. - minmax[:, 1] * x
    # w = w * x
    # b = np.dot(w, y) + b

    return w, b