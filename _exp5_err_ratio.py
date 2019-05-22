import numpy as np
import neural_network as nn
from time import time
from data_loader import load_wine
from chart_utils import surface_chart
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start = time()

    ile = 10
    ratios = np.arange(1.01, 1.5, 0.01)
    results = np.empty((ile, len(ratios)))

    for j in range(ile):
        for i, err_ratio in enumerate(ratios):
            while True:
                learning_data, testing_data = load_wine()
                net = nn.NeuralNetwork(0.01, 100, [5, 4], err_ratio, 1.2, 0.6, 0.020)       # int((s[0]*s[1])*5)
                net.feed_training_data(*learning_data)
                net.feed_test_data(*testing_data)
                net.start_learning()
                pk = net.test(*testing_data)[2]
                if pk > 20:
                    results[j][i] = pk
                    break

            print(f'[{(j*len(ratios))+i+1:2d} / {len(ratios)*ile:2d}]  {err_ratio:3.2f}  {pk:3.0f}% PK')

    end = time()
    print(f'Time: {end-start} sec')

    results = np.average(results, axis=0)

    with open(f'data/err_ratios.csv', 'w') as f:
        for x, y in zip(ratios, results):
            f.write(f'{x:3.2f};{y:3.0f}\n')

    # surface_chart('data/ada_lr.csv', 'lr_inc', 'lr_dec', '% PK')


    plt.plot(ratios, results)
    plt.ylim(0, 100)
    plt.show()
