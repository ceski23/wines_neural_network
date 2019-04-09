import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn
from random import randrange

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
# np.seterr('raise')

def load_wine(test_count):
    with open('wine.csv') as f:
        wines = np.array([list(map(float, x.strip().split(','))) for x in f])
        np.random.shuffle(wines)
        a = int((wines.shape[0] / 100) * test_count)
        wines[:, 1:] = normalize(wines[:, 1:].T).T

        test_wines = wines[:a]
        test_wines = test_wines[test_wines[:,0].argsort()]
        testing_data = (test_wines[:, 1:], test_wines[:, 0])

        wines = wines[a:]
        wines = wines[wines[:,0].argsort()]
        learning_data = (wines[:, 1:], wines[:, 0])
    return (learning_data, testing_data)

def normalize(data, min_v=0, max_v=1):
    '''Normalizuje dane do podanego zakresu'''
    for i in range(data.shape[0]):
        data[i] = ((data[i] - data[i].min()) / (data[i].max() - data[i].min())) * (max_v - min_v) + min_v
    return data


if __name__ == "__main__":
    learning_data, testing_data = load_wine(test_count=20)
    
    net = nn.NeuralNetwork(0.2, 20, [26, 12, 4], 1.04, 1.05, 0.7, 0.020)
    net.feed_training_data(*learning_data)
    net.feed_test_data(*testing_data)
    net.start_learning(live_plot=False)
    net.save_model()


    plt.figure()
    for i in range(4):
        P, T = load_wine(test_count=20)[1]
        prediction = [net.predict(x) for x in P]
        error = [d-y for y, d in zip(prediction, T)]
        cost = [0.5 * (e**2) for e in error]

        plt.subplot(2, 2, i+1)
        plt.grid(linestyle='--')
        plt.yticks(np.arange(min(T), max(T)+0.01, 0.25))
        plt.plot(prediction)
        plt.plot(T)
        plt.title(f'MAX cost: {max(cost):.8f}')
    plt.show()