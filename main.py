import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn
from random import randrange

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
# np.seterr('raise')

def load_zoo(test_count):
    with open('zoo.csv') as f:
        animals = np.array([list(map(int, x.strip().split(',')[1:])) for x in f])
        np.random.shuffle(animals)
        a = int((animals.shape[0] / 100) * test_count)

        test = animals[:a]
        test = test[test[:,16].argsort()]
        test_data = (normalize(test[:, :16].T).T, test[:, 16])

        learning = animals[a:]
        learning = learning[learning[:,16].argsort()]
        learning_data = (normalize(learning[:, :16].T).T, learning[:, 16])
    return [learning_data, test_data]

def load_wine(test_count):
    with open('wine.csv') as f:
        wines = np.array([list(map(float, x.strip().split(','))) for x in f])
        np.random.shuffle(wines)
        a = int((wines.shape[0] / 100) * test_count)
        wines[:, 1:] = normalize(wines[:, 1:].T).T

        # test_wines = wines[:a]
        # test_wines = test_wines[test_wines[:,0].argsort()]
        # testing_data = (normalize(test_wines[:, 1:].T).T, test_wines[:, 0])

        # wines = wines[a:]
        # wines = wines[wines[:,0].argsort()]
        # learning_data = (normalize(wines[:, 1:].T).T, wines[:, 0])

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
    
    net = nn.NeuralNetwork(0.2, 2000, [26, 12, 4], 1.04, 1.05, 0.7, 0.020)
    # net.feed_training_data(*learning_data)
    # net.feed_test_data(*testing_data)
    # net.start_learning(live_plot=True)
    # net.save_model('wine.model')
    net.load_model('wine/001900_26_12_4_02.model')



    P, T = testing_data
    prediction = [net.predict(x) for x in P]
    loss = [net.loss(z, y) for z, y in zip(prediction, T)]
    mse = [0.5 * (l**2) for l in loss]

    plt.figure()
    plt.grid(linestyle='--')
    plt.yticks(np.arange(min(T), max(T), 0.25))
    plt.plot(prediction)
    plt.plot(T)
    plt.title(f'MAX MSE: {max(mse)} \nAVG MSE: {sum(mse)/len(mse)}')
    plt.show()