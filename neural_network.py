import matplotlib.pyplot as plt
import numpy as np
from time import time
import pickle


class NeuralNetwork:
    def __init__(self, n, lr, epoch_n):
        self.n = n
        self.lr = lr
        self.epoch_n = epoch_n
    
    def feed_training_data(self, P, T):
        self.P = P
        self.T = T
    
    def sigmoid(self, x, derivative=False):
        '''Zwraca wartość funkcji sigmoidalnej dla danego x'''
        if derivative:
            f = self.sigmoid(x)
            return f * (1 - f)
        else:
            return 1 / (1 + (np.e**(-x)))
    
    def linear(self, x, derivative=False):
        return 1 if derivative else x

    def neuron(self, x, w, func):
        '''Oblicza wyjście dla pojedynczego neuronu'''
        e = sum(x * w)
        return (func(e), e)

    def loss(self, z, y):
        '''Oblicza błąd w danej serii danych'''
        return z - y

    def calc_neuron_loss(self, l, w):
        '''Propagacja błędu w jednej warstwie'''
        return l * w

    def update_weights(self, x, w, func, l, e):
        '''Obliczanie wag dla pojedynczego neuronu'''
        d = self.lr * l * func(e, True) * x
        return w + d
    
    def predict(self, P):
        y = np.array([self.neuron(P, w, self.sigmoid)[0] for w in self.weights_1])
        Y = self.neuron(y, self.weights_2, self.linear)[0]
        return Y
    
    def start_learning(self, live_plot=False):
        if live_plot:
            plt.plot(self.T)
            axes = plt.gca()
            line, = axes.plot([], [], 'r-')
            line.set_xdata(range(self.T.shape[0]))

        self.weights_1 = np.random.rand(self.n, self.P.shape[1])
        self.weights_2 = np.random.rand(self.n)

        for epoch in range(self.epoch_n):
            epoch_loss, prediction = [], []
            for x, z in zip(self.P, self.T):
                ey, y = [], []
                for w in self.weights_1:
                    q = self.neuron(x, w, self.sigmoid)
                    y.append(q[0])
                    ey.append(q[1])
                # ey = np.array(ey)
                y = np.array(y)
                # y = np.array([self.neuron(x, w, self.sigmoid) for w in self.weights_1])
                Y, eY = self.neuron(y, self.weights_2, self.linear)

                L = self.loss(z, Y)
                l1 = self.calc_neuron_loss(L, self.weights_2)

                self.weights_1 = np.array([self.update_weights(x, self.weights_1[j], self.sigmoid, l1[j], ey[j]) for j in range(self.n)])
                self.weights_2 = self.update_weights(y, self.weights_2, self.linear, L, eY)

                epoch_loss.append(L)
                prediction.append(Y)

            stats = np.array(epoch_loss)
            if live_plot:
                line.set_ydata(prediction)
                axes.set_title(f'Epoka #{epoch}\nSTD: {stats.std():.10f}   MAX: {stats.max():.10f}')
                plt.draw()
                plt.pause(1e-17)
            else:
                print(f'Epoka #{epoch:02d} ({stats.max():.10f})', end='\r')
        
        plt.show()

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.weights_1, self.weights_2), f)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.weights_1, self.weights_2 = pickle.load(f)