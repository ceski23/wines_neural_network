import numpy as np
import neural_network as nn
from time import time
from data_loader import load_wine
from chart_utils import surface_chart


if __name__ == "__main__":
    # a, b = range(10, 101, 20), range(10, 101, 20)
    a, b = range(2, 10), range(2, 10)
    S1, S2 = np.meshgrid(a, b)
    c, t = 1, len(a) * len(b)
    PK = np.empty(S1.shape, dtype=int)

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('%PK')
    ax.set_zlim(0, 100)

    learning_data, testing_data = load_wine(test_count=20)

    start = time()

    for i, row in enumerate(zip(S1, S2)):
        for j, s in enumerate(zip(*row)):
            while True:
                learning_data, testing_data = learning_data.copy(), testing_data.copy()
                net = nn.NeuralNetwork(0.01, int(s[0]*s[1]/4), [*s], 1.04, 1.05, 0.7, 0.020)
                net.feed_training_data(*learning_data)
                net.feed_test_data(*testing_data)
                net.start_learning(live_text=True)
                prediction = net.test(*testing_data)[0]
                pk = [abs(x - y) < 0.25 for x, y in zip(prediction, testing_data[1])]
                result = int((sum(pk) / len(pk)) * 100)
                if result > 10:
                    break
            
            PK[i][j] = result
            print(f'[{c:2d} / {t:2d}] {s} {result:3d}% PK    Pozostało: {t-c}                                        ')
            c += 1

    end = time()
    print(f'Single-core time: {end-start} sec')
    
    with open('neurons_chart_data.csv', 'w') as f:
        for x, y, z in zip(S1.flatten(), S2.flatten(), PK.flatten()):
            f.write(f'{x};{y};{z}\n')