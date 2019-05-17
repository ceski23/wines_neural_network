import numpy as np
import neural_network as nn
from time import time
import multiprocessing as mp
from data_loader import load_wine
from chart_utils import surface_chart
import matplotlib.pyplot as plt


def chart_job(s, i, j):
    while True:
        learning_data, testing_data = load_wine()
        net = nn.NeuralNetwork(0.01, 100, [5, 4], 1.04, *s, 0.020)       # int((s[0]*s[1])*5)
        net.feed_training_data(*learning_data)
        net.feed_test_data(*testing_data)
        net.start_learning()
        prediction, cost, pk = net.test(*testing_data)

        if cost < 15 and cost > 0:
            break

    return (i, j, (cost, pk), s)


    # return (i, j, result, s)


def chart_job_callback(x):
    global c
    result[x[0]][x[1]] = x[2][0]
    result2[x[0]][x[1]] = x[2][1]
    print(f'[{c:2d} / {t:2d}]  ({x[3][0]:3.2f}, {x[3][1]:3.2f})  {int(result2[x[0]][x[1]]):3d}%PK    Pozostało: {t-c}')
    c += 1


if __name__ == "__main__":
    start = time()
    for x in range(1, 2):
        a, b = np.arange(1.00, 1.91, 0.1), np.arange(0.5, 1.00, 0.05)

        lr_inc, lr_dec = np.meshgrid(a, b)
        c, t = 1, len(a) * len(b)
        result = np.empty(lr_dec.shape)
        result2 = np.empty(lr_inc.shape)
        pool = mp.Pool(7)

        for i, row in enumerate(zip(lr_inc, lr_dec)):
            for j, s in enumerate(zip(*row)):
                # chart_job_callback(chart_job(s, i, j))
                pool.apply_async(chart_job, 
                    args=(s, i, j),
                    callback=chart_job_callback)
        
        pool.close()
        pool.join()

        with open(f'p{x}.csv', 'w') as f:
            for x, y, z, pk in zip(lr_inc.flatten(), lr_dec.flatten(), result.flatten(), result2.flatten()):
                f.write(f'{x:3.2f};{y:3.2f};{int(pk)}\n')

    end = time()
    print(f'Multi-core time: {end-start} sec')
    surface_chart('p1.csv', 'lr_inc', 'lr_dec', '% PK')
    plt.show()