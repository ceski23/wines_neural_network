import matplotlib.pyplot as plt
import numpy as np
from time import time

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
np.random.seed(123)


def load_data(path):
    with open(path) as f:
        wines = []
        for x in f:
            x = x.strip().split(',')
            wines.append(list(map(float, x)))
        return np.array(wines)

def normalize(data):
    '''Normalizuje dane do zakresu [0-1]'''
    return ((data - data.min()) / (data.max() - data.min()))


def sigmoid(x):
    '''Zwraca wartość funkcji sigmoidalnej dla danego x'''
    return 1 / (1 + (np.e**(-x)))

def sigmoid_der(x):
    '''Zwraca wartość pochodnej funkcji sigmoidalnej dla danego x'''
    f = sigmoid(x)
    return f * (1 - f)

def neuron(x, w, func):
    '''Oblicza wyjście dla pojedynczego neuronu'''
    e = sum(x * w)
    return (e, func(e))

def loss(z, y):
    '''Oblicza błąd w danej serii danych'''
    return z - y

def calc_neuron_loss(l, w):
    '''Propagacja błędu w jednej warstwie'''
    return l * w

def update_weights(x, w, func, l, e):
    '''Obliczanie wag dla pojedynczego neuronu'''
    d = lr * l * func(e) * x
    # d = lr * l * func(np.sum(x * w)) * x
    return w + d




n = 200  # liczba neuronów
lr = 0.1   # learning rate
epoch_n = 1000   # liczba epok

wines = load_data('wine.csv')
P = normalize(wines[:, 1:])
T = normalize(wines[:, 0])
w1 = np.random.rand(n, P.shape[1])
w2 = np.random.rand(n)

start_time = time()

# score = []
s_max, s_std = [], []
for i in range(epoch_n):
    predicted, ep_l = [], []
    for x, z in zip(P, T):
        ey, y = [], []
        for w in w1:
            a, b = neuron(x, w, sigmoid)
            ey.append(a)
            y.append(b)
        y = np.array(y)
        ey = np.array(ey)

        # y = np.fromiter((neuron(x, w, sigmoid) for w in w1), float)
        # ey = y[:,0]
        # t = y[:,1]

        eY, Y = neuron(y, w2, sigmoid)

        L = loss(z, Y)
        l1 = calc_neuron_loss(L, w2)

        w1 = np.array([update_weights(x, w1[j], sigmoid_der, l1[j], y[j]) for j in range(n)])
        # w1 = np.array([update_weights(x, w1[j], sigmoid_der, l1[j], ey[j]) for j in range(n)])
        w2 = update_weights(y, w2, sigmoid_der, L, Y)
        # w2 = update_weights(y, w2, sigmoid_der, L, eY)

        predicted.append(Y)
        ep_l.append(L)

    # q = np.array(predicted - T)
    # score.append((sum(abs(q) <= 0.5) / len(P)) * 100)
    s = np.array(ep_l)
    s_max.append(np.abs(s).max())
    s_std.append(s.std())
    print(f'Epoka #{i:02d} ({np.abs(s).max():.10f}) [{s.std():.10f}]', end='\r')


print(f'\nCzas uczenia: {(time()-start_time)/1000} sekund')

plt.plot(T)
plt.plot(predicted)

plt.figure()
plt.plot(s_max)
plt.plot(s_std)
# plt.plot(score)

plt.show()