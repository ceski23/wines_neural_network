import matplotlib.pyplot as plt
import numpy as np
import neural_network as nn


if __name__ == "__main__":
    net = nn.NeuralNetwork(None, None, None, None, None, None, None)
    net.load_model('models/004405_16_12.mdl')

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