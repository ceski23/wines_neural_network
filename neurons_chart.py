import numpy as np
import neural_network as nn
from time import time
import multiprocessing as mp
from data_loader import load_wine
from chart_utils import surface_chart2


def chart_job(s, i, j):
    while True:
        learning_data, testing_data = load_wine()
        net = nn.NeuralNetwork(0.01, 1000, [*s], 1.04, 1.05, 0.7, 0.020)       # int((s[0]*s[1])*5)
        net.feed_training_data(*learning_data)
        net.feed_test_data(*testing_data)
        net.start_learning()
        prediction, cost, pk = net.test(*testing_data)

        if cost < 15:
            break

    return (i, j, (cost, pk), s)


    # return (i, j, result, s)


def chart_job_callback(x):
    global c
    result[x[0]][x[1]] = x[2][0]
    result2[x[0]][x[1]] = x[2][1]
    print(f'[{c:2d} / {t:2d}] {x[3]} {result[x[0]][x[1]]:7.4f}    Pozostało: {t-c}')
    c += 1


if __name__ == "__main__":
    a, b = range(1, 101, 10), range(1, 101, 10)

    S1, S2 = np.meshgrid(a, b)
    c, t = 1, len(a) * len(b)
    result = np.empty(S1.shape)
    result2 = np.empty(S1.shape)
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
        for x, y, z, pk in zip(S1.flatten(), S2.flatten(), result.flatten(), result2.flatten()):
            f.write(f'{x};{y};{z};{pk}\n')

    surface_chart2('neurons_chart_data.csv', 'S1', 'S2', 'Cost', '% PK')