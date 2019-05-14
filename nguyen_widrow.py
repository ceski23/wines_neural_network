import numpy as np

def initnw(layer_size, input_size):
    beta = 0.7 * (layer_size ** (1.0 / input_size))
    w_rand = np.random.rand(layer_size, input_size) * 2 - 1
    w_rand = np.sqrt(1.0 / np.square(w_rand).sum(axis=1).reshape(layer_size, 1)) * w_rand

    w = beta * w_rand
    b = beta * np.linspace(-1, 1, layer_size) * np.sign(w[:, 0])

    return w, b