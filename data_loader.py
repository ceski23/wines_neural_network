import numpy as np


def normalize(data, min_v=-1, max_v=1):
    '''Normalizuje dane do podanego zakresu'''
    for i in range(data.shape[0]):
        data[i] = ((data[i] - data[i].min()) / (data[i].max() - data[i].min())) * (max_v - min_v) + min_v
    return data


def load_zoo(test_count):
    with open('zoo.csv') as f:
        animals = np.array([list(map(float, x.strip().split(',')[1:])) for x in f])
        np.random.shuffle(animals)
        a = int((animals.shape[0] / 100) * test_count)

        test = animals[:a]
        test = test[test[:,16].argsort()]
        test_data = (normalize(test[:, :16].T).T, test[:, 16])

        learning = animals[a:]
        learning = learning[learning[:,16].argsort()]
        learning_data = (normalize(learning[:, :16].T).T, learning[:, 16])
    return [learning_data, test_data]


def load_wine(test_count=20):
    with open('wine.csv') as f:
        wines = np.array([list(map(float, x.strip().split(','))) for x in f])
        np.random.shuffle(wines)
        a = int((wines.shape[0] / 100) * test_count)
        wines[:, 1:] = normalize(wines[:, 1:].T).T

        test_wines = wines[:a]
        test_wines = test_wines[test_wines[:,0].argsort()]
        testing_data = (test_wines[:, 1:], test_wines[:, 0])

        wines = wines[a:]
        # wines = wines[wines[:,0].argsort()]
        learning_data = (wines[:, 1:], wines[:, 0])
    
    return (learning_data, testing_data)