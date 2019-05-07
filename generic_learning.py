import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn
from time import time
from data_loader import load_wine, load_zoo


if __name__ == "__main__":
    learning_data, testing_data = load_wine(test_count=20)
    
    net = nn.NeuralNetwork(0.01, 10000, [10, 6], 1.04, 1.05, 0.7, 0.020)
    net.feed_training_data(*learning_data)
    net.feed_test_data(*testing_data)
    s = time()
    net.start_learning(live_text=True, live_plot=False, plot_results=True)
    e = time()
    print(f'\nTime: {e-s} sec')
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