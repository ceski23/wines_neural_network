import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import neural_network as nn
from time import time
import multiprocessing as mp
from data_loader import load_wine


def chart_job(s, i, j, learning_data, testing_data):
    test = False
    while not test:
        net = nn.NeuralNetwork(0.08, 1000, [*s], 1.04, 1.05, 0.7, 0.020)
        net.feed_training_data(*learning_data)
        net.feed_test_data(*testing_data)
        net.start_learning()
        prediction = net.test(*testing_data)[0]
        pk = [abs(x - y) < 0.25 for x, y in zip(prediction, testing_data[1])]
        result = int((sum(pk) / len(pk)) * 100)
        test = True# if result > 10 else False
    return (i, j, result)


def chart_job_callback(x):
    global c
    PK[x[0]][x[1]] = x[2]
    print(f'[{c:2d} / {t:2d}] {PK[x[0]][x[1]]:3d}% PK    Pozostało: {t-c}')
    c += 1


if __name__ == "__main__":
    a, b = range(10, 101, 20), range(10, 101, 20)
    S1, S2 = np.meshgrid(a, b)
    c, t = 1, len(a) * len(b)
    PK = np.empty(S1.shape, dtype=int)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('%PK')
    ax.set_zlim(0, 100)

    pool = mp.Pool(7)
    learning_data, testing_data = load_wine(test_count=20)

    start = time()

    for i, row in enumerate(zip(S1, S2)):
        for j, s in enumerate(zip(*row)):
            pool.apply_async(chart_job, 
                args=(s, i, j, learning_data, testing_data), 
                callback=chart_job_callback)
    
    pool.close()
    pool.join()

    end = time()
    print(f'Multi-core time: {end-start} sec')

    ax.plot_surface(S1, S2, np.array(PK), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()