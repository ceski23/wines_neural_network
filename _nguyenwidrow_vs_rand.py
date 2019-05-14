import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn
from data_loader import load_wine


if __name__ == "__main__":
    learning_data, testing_data = load_wine()
    net = nn.NeuralNetwork(0.01, 20, [8, 2], 1.04, 1.05, 0.7, 0.20)
    # net = nn.NeuralNetwork(0.01, 10000, [20, 10], 1.04, 1.05, 0.7, 0.020)
    net.feed_training_data(*learning_data)
    net.feed_test_data(*testing_data)
    net.start_learning(live_text=True, live_plot=False, plot_results=True)

    plt.show()