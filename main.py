import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn
from random import randrange

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
np.seterr('raise')
# np.random.seed(123)

def load_zoo():
    with open('zoo.csv') as f:
        animals = np.array([list(map(int, x.strip().split(',')[1:])) for x in f])
        animals = animals[animals[:,16].argsort()]
        P = normalize(animals[:, :-1])
        T = animals[:, -1]
    return (P, T)

def load_wine(test_perc):
    with open('wine.csv') as f:
        wines = np.array([list(map(float, x.strip().split(','))) for x in f])
        np.random.shuffle(wines)
        a = int((wines.shape[0] / 100) * test_perc)

        test_wines = wines[:a]
        test_wines = test_wines[test_wines[:,1].argsort()]
        testing_data = (normalize(test_wines[:, 1:]), test_wines[:, 0])

        wines = wines[a:]
        wines = wines[wines[:,1].argsort()]
        learning_data = (normalize(wines[:, 1:]), wines[:, 0])
    return (learning_data, testing_data)

def normalize(data, min_v=0, max_v=1):
    '''Normalizuje dane do podanego zakresu'''
    return ((data - data.min()) / (data.max() - data.min())) * (max_v - min_v) + min_v


if __name__ == "__main__":
    learning_data, testing_data = load_wine(test_perc=20)
    
    net = nn.NeuralNetwork(n=26, lr=0.5, epoch_n=200, hidden_size=12)
    net.feed_training_data(*learning_data)
    net.start_learning(live_plot=True)
    net.save_model('wine.model')
    # net.load_model('wine.model')

    # y = [net.predict(x) for x in P]
    # plt.plot(y)
    # plt.plot(T)
    # plt.show()


# Potencjał:
# 20, 0.1, 1000, 20
# 20, 0.5, 1000, 10