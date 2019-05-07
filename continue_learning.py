import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn
from data_loader import load_wine
from time import time


if __name__ == "__main__":
    learning_data, testing_data = load_wine(test_count=20)
    net = nn.NeuralNetwork(None, 10000, None, 1.04, 1.05, 0.7, 0.0005)
    net.feed_training_data(*learning_data)
    net.feed_test_data(*testing_data)
    net.load_model('models/000100_10_6.mdl')

    s = time()
    net.start_learning(live_text=True, live_plot=False, plot_results=True)
    e = time()

    print(f'\nTime: {e-s} sec')
    net.save_model()