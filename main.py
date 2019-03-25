import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn
from random import randrange

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
np.seterr('raise')
np.random.seed(123)

def load_zoo():
    with open('zoo.csv') as f:
        animals = np.array([list(map(int, x.strip().split(',')[1:])) for x in f])
        animals = animals[animals[:,16].argsort()]
        P = normalize(animals[:, :-1])
        T = animals[:, -1]
    return (P, T)

def load_wine():
    with open('wine.csv') as f:
        wines = []
        for x in f:
            x = x.strip().split(',')
            wines.append(list(map(float, x)))
        wines = np.array(wines)
        P = normalize(wines[:, 1:])
        T = wines[:, 0]
    return (P, T)

def normalize(data, min_v=0, max_v=1):
    '''Normalizuje dane do podanego zakresu'''
    return ((data - data.min()) / (data.max() - data.min())) * (max_v - min_v) + min_v


if __name__ == "__main__":
    P, T = load_wine()
    
    # idxs = list(set(randrange(0, P.shape[0]) for _ in range(21)))
    # X, Y = P[idxs], T[idxs]
    # P = np.delete(P, idxs, 0)
    # T = np.delete(T, idxs)
    
    net = nn.NeuralNetwork(n=20, lr=0.3, epoch_n=12)
    net.feed_training_data(P, T)
    net.start_learning(live_plot=True)
    net.save_model('wine.model')

    # y = [net.predict(x) for x in X]
    # plt.plot(y)
    # plt.plot(Y)
    # plt.show()