import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn
from data_loader import load_wine


if __name__ == "__main__":
    net = nn.NeuralNetwork(None, None, None, None, None, None, None)
    net.load_model('models/000010_10_6.mdl')

    plt.figure()
    for i in range(12):
        P, T = load_wine()[1]
        prediction, cost, pk = net.test(P, T)

        plt.subplot(3, 4, i+1)
        plt.grid(linestyle='--')
        plt.yticks(np.arange(min(T), max(T)+0.01, 0.25))
        plt.plot(prediction)
        plt.plot(T)
        plt.title(f'Cost: {cost:.8f}  PK: {pk}%')
    plt.show()