import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import neural_network as nn
from time import time
import multiprocessing as mp

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


def neurons_amount_chart(a, b):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('%PK')
    ax.set_zlim(0, 100)

    S1, S2 = np.meshgrid(range(*a), range(*b))
    PK, i, t = [], 1, len(range(*a)) * len(range(*b))
    for s1_r, s2_r in zip(S1, S2):
        z = []
        for s1, s2 in zip(s1_r, s2_r):
            net = nn.NeuralNetwork(0.1, 1000, [s1, s2], 1.04, 1.05, 0.7, 0.020)
            learning_data, testing_data = load_wine(test_count=20)
            net.feed_training_data(*learning_data)
            net.feed_test_data(*testing_data)
            net.start_learning()
            prediction = net.test(*testing_data)[0]
            pk = [abs(x - y) < 0.25 for x, y in zip(prediction, testing_data[1])]
            z.append(int((sum(pk) / len(pk)) * 100))
            print(f'[{i:2d} / {t:2d}] {z[-1]:3d}% PK', end='\r')
            i += 1
        PK.append(z)

    ax.plot_surface(S1, S2, np.array(PK), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()


def generic_learning():
    learning_data, testing_data = load_wine(test_count=20)
    
    net = nn.NeuralNetwork(0.1, 20000, [18, 7], 1.04, 1.05, 0.7, 0.020)
    net.feed_training_data(*learning_data)
    net.feed_test_data(*testing_data)
    s = time()
    net.start_learning(live_text=True, plot_results=True)
    e = time()
    print(f'Time: {e-s} sec')
    net.save_model()


    plt.figure()
    for i in range(12):
        P, T = load_wine(test_count=20)[1]
        prediction, cost = net.test(P, T)

        plt.subplot(3, 4, i+1)
        plt.grid(linestyle='--')
        plt.yticks(np.arange(min(T), max(T)+0.01, 0.25))
        plt.plot(prediction)
        plt.plot(T)
        plt.title(f'Cost: {cost:.8f}')
    plt.show()




def raw_learn(ns):
        learning_data, testing_data = load_wine(test_count=20)
        net = nn.NeuralNetwork(0.1, 100, ns, 1.04, 1.05, 0.7, 0.020)
        net.feed_training_data(*learning_data)
        net.feed_test_data(*testing_data)
        net.start_learning()

def multicore_learning():
    s = time()
    for i in range(16, 24):
        raw_learn([i, 7])
    e = time()
    print(f'Single-core time: {e-s} sec')


    pool = mp.Pool()
    s = time()
    for i in range(16, 24):
        pool.apply_async(raw_learn, args=([i, 7], ))
    pool.close()
    pool.join()
    e = time()
    print(f'Multi-core time: {e-s} sec')




def chart_job(s, i, j):
    test = False
    while not test:
        net = nn.NeuralNetwork(0.05, 10000, [*s], 1.04, 1.05, 0.7, 0.020)
        learning_data, testing_data = load_wine(test_count=20)
        net.feed_training_data(*learning_data)
        net.feed_test_data(*testing_data)
        net.start_learning()
        prediction = net.test(*testing_data)[0]
        pk = [abs(x - y) < 0.25 for x, y in zip(prediction, testing_data[1])]
        result = int((sum(pk) / len(pk)) * 100)
        test = True if result > 50 else False
    return (i, j, result)

def chart_job_callback(x):
    global c
    PK[x[0]][x[1]] = x[2]
    print(f'[{c:2d} / {t:2d}] {PK[x[0]][x[1]]:3d}% PK    Pozostało: {t-c}')
    c += 1

def multicore_chart(a, b):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('%PK')
    ax.set_zlim(0, 100)

    S1, S2 = np.meshgrid(range(*a), range(*b))
    c, t = 1, len(range(*a)) * len(range(*b))
    PK = np.empty(S1.shape, dtype=int)
    pool = mp.Pool()

    start = time()

    for i, row in enumerate(zip(S1, S2)):
        for j, s in enumerate(zip(*row)):
            pool.apply_async(chart_job, args=(s, i, j, ), callback=chart_job_callback)
    
    pool.close()
    pool.join()

    end = time()
    print(f'Multi-core time: {end-start} sec')

    ax.plot_surface(S1, S2, np.array(PK), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()




def test_existing_model(path):
    net = nn.NeuralNetwork(None, None, None, None, None, None, None)
    net.load_model(path)

    plt.figure()
    for i in range(12):
        P, T = load_wine(test_count=20)[1]
        prediction, cost = net.test(P, T)

        plt.subplot(3, 4, i+1)
        plt.grid(linestyle='--')
        plt.yticks(np.arange(min(T), max(T)+0.01, 0.25))
        plt.plot(prediction)
        plt.plot(T)
        plt.title(f'Cost: {cost:.8f}')
    plt.show()


if __name__ == "__main__":
    a, b = [10, 20], [3, 8]
    S1, S2 = np.meshgrid(range(*a), range(*b))
    c, t = 1, len(range(*a)) * len(range(*b))
    PK = np.empty(S1.shape, dtype=int)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('%PK')
    ax.set_zlim(0, 100)

    pool = mp.Pool()

    start = time()

    for i, row in enumerate(zip(S1, S2)):
        for j, s in enumerate(zip(*row)):
            pool.apply_async(chart_job, args=(s, i, j, ), callback=chart_job_callback)
    
    pool.close()
    pool.join()

    end = time()
    print(f'Multi-core time: {end-start} sec')

    ax.plot_surface(S1, S2, np.array(PK), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()













    # generic_learning()
    # multicore_learning()
    # multicore_chart([10, 20], [3, 8])
    # test_existing_model('models/004405_16_12.mdl')
    # neurons_amount_chart([10, 20], [3, 8])

    





