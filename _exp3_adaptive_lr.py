import numpy as np
import neural_network as nn
from time import time
from data_loader import load_wine
from chart_utils import surface_chart
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start = time()

    a, b = np.arange(1.00, 1.91, 0.1), np.arange(0.5, 1.00, 0.05)
    lr_inc, lr_dec = np.meshgrid(a, b)
    result = np.empty(lr_dec.shape)

    for i, row in enumerate(zip(lr_inc, lr_dec)):
        for j, params in enumerate(zip(*row)):
            learning_data, testing_data = load_wine()
            net = nn.NeuralNetwork(0.01, 100, [5, 4], 1.04, *params, 0.020)       # int((s[0]*s[1])*5)
            net.feed_training_data(*learning_data)
            net.feed_test_data(*testing_data)
            net.start_learning()
            pk = net.test(*testing_data)[2]
            result[i][j] = pk

            print(f'[{len(lr_inc)*i+j+1:2d} / {len(a)*len(b):2d}]  ({params[0]:3.2f}, {params[1]:3.2f})  {pk:3.0f}% PK')

    end = time()
    print(f'Time: {end-start} sec')

    with open(f'data/ada_lr.csv', 'w') as f:
        for x, y, z in zip(lr_inc.flatten(), lr_dec.flatten(), result.flatten()):
            f.write(f'{x:3.2f};{y:3.2f};{z:3.0f}\n')

    surface_chart('data/ada_lr.csv', 'lr_inc', 'lr_dec', '% PK')
    plt.show()
