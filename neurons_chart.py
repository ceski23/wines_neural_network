import numpy as np
import neural_network as nn
from time import time
import multiprocessing as mp
from data_loader import load_wine
from chart_utils import surface_chart


def chart_job(s, i, j):
    learning_data, testing_data = load_wine(test_count=20)
    net = nn.NeuralNetwork(0.01, 100, [*s], 1.04, 1.05, 0.7, 0.020)       # int((s[0]*s[1])*5)
    net.feed_training_data(*learning_data)
    net.feed_test_data(*testing_data)
    net.start_learning()
    prediction, cost = net.test(*testing_data)

    return (i, j, cost, s)

    # pk = [abs(x - y) < 0.25 for x, y in zip(prediction, testing_data[1])]
    # result = int((sum(pk) / len(pk)) * 100)
    # return (i, j, result, s)


def chart_job_callback(x):
    global c
    result[x[0]][x[1]] = x[2]
    print(f'[{c:2d} / {t:2d}] {x[3]} {result[x[0]][x[1]]:7.4f}    Pozostało: {t-c}')
    c += 1


if __name__ == "__main__":
    a, b = range(1, 10), range(1, 10)

    S1, S2 = np.meshgrid(a, b)
    c, t = 1, len(a) * len(b)
    result = np.empty(S1.shape)
    pool = mp.Pool(7)

    start = time()

    for i, row in enumerate(zip(S1, S2)):
        for j, s in enumerate(zip(*row)):
            # chart_job_callback(chart_job(s, i, j))
            pool.apply_async(chart_job, 
                args=(s, i, j),
                callback=chart_job_callback)
    
    pool.close()
    pool.join()

    end = time()
    print(f'Multi-core time: {end-start} sec')

    with open('neurons_chart_data.csv', 'w') as f:
        for x, y, z in zip(S1.flatten(), S2.flatten(), result.flatten()):
            f.write(f'{x};{y};{z}\n')

    surface_chart('neurons_chart_data.csv', 'S1', 'S2', 'Cost')