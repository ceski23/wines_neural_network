import matplotlib.pyplot as plt
import numpy as np
from time import time
import pickle
import math


class NeuralNetwork:
    def __init__(self, lr, epoch_n, hidden, err_ratio, lr_inc, lr_dec, goal):
        self.lr = lr
        self.epoch_n = epoch_n
        self.hidden = hidden
        self.err_ratio = err_ratio
        self.lr_inc = lr_inc
        self.lr_dec = lr_dec
        self.goal = goal
    
    def feed_training_data(self, P, T):
        self.P = P
        self.T = T
    
    def feed_test_data(self, P, T):
        self.test_P = P
        self.test_T = T

    def sigmoid(self, x, derivative=False):
        '''Zwraca wartość funkcji sigmoidalnej dla danego x'''
        if derivative:
            f = self.sigmoid(x)
            return f * (1 - f)
        else:
            return 1 / (1 + (np.e**(-x)))
    
    def linear(self, x, derivative=False):
        return 1 if derivative else x

    def neuron(self, x, w):
        '''Oblicza wyjście dla pojedynczego neuronu'''
        return sum(x * w)

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
    
    def init_weights(self):
        self.weights = []

        sizes = [self.P.shape[1], *self.hidden, 1]
        for i in range(1, len(sizes)):
            self.weights.append(np.random.rand(sizes[i], sizes[i-1]))
    
    def predict(self, x):
        y = [x]
        for i in range(len(self.weights)):
            func = self.linear if i == len(self.weights) - 1 else self.sigmoid
            out_e = np.array([self.neuron(y[i], w) for w in self.weights[i]])
            y.append(func(out_e))

        return y[-1][0]
    
    def start_learning(self, live_plot=False, plot_interval=1):
        if live_plot:
            plt.plot(self.test_T)
            plt.grid(linestyle='--')
            axes = plt.gca()
            axes.set_yticks(np.arange(min(self.test_T), max(self.test_T), 0.25), minor=True)
            line, = axes.plot([], [], 'r-')
            line.set_xdata(range(self.test_T.shape[0]))

        self.init_weights()

        mses, last_weights = [], []
        backup = {'weights': None, 'mse': math.inf, 'epoch': -1}

        for epoch in range(self.epoch_n):
            for x, z in zip(self.P, self.T):
                y, e = [x], []

                for i in range(len(self.weights)):
                    func = self.linear if i == len(self.weights) - 1 else self.sigmoid
                    out_e = np.array([self.neuron(y[i], w) for w in self.weights[i]])
                    y.append(func(out_e))
                    e.append(out_e)

                errors = [self.loss(z, y[-1])]
                for layer_idx in range(len(self.weights)-1, 0, -1):
                    layer_error = [sum(self.calc_neuron_loss(errors[0], w)) for w in self.weights[layer_idx].T]
                    errors.insert(0, layer_error)

                for i in range(len(self.weights)):
                    func = self.linear if i == len(self.weights) - 1 else self.sigmoid
                    for j in range(len(self.weights[i])):
                        self.weights[i][j] = self.update_weights(y[i], self.weights[i][j], func, errors[i][j], e[i][j])


            prediction = [self.predict(x) for x in self.test_P]
            loss = [self.loss(z, y) for z, y in zip(prediction, self.test_T)]
            mse = max([0.5 * (l**2) for l in loss])
            mses.append(mse)

            if mse <= self.goal:
                print(f'Achieved goal with MSE: {mse}')
                break

            if len(mses) >= 2:
                if mses[-1] > mses[-2]*self.err_ratio:
                    self.lr *= self.lr_dec
                    self.weights = last_weights
                elif mses[-1] < mses[-2]: #*(2 - self.err_ratio):
                    self.lr *= self.lr_inc

            last_weights = self.weights

            if mse < backup['mse']:
                backup['mse'] = mse
                backup['weights'] = self.weights
                backup['epoch'] = epoch

            if live_plot and not epoch % plot_interval:
                line.set_ydata(prediction)
                axes.set_title(f'Epoka #{epoch}\nMSE: {mse:2.10f} \n LR: {self.lr}')
                plt.draw()
                plt.pause(1e-20)
            
            if not live_plot:
                print(f'Epoka #{epoch:02d} MSE: {mse:2.10f}', end='\r')
        
        if not live_plot:
            plt.plot(prediction)
            plt.plot(self.test_T)

        plt.figure()
        plt.plot(mses)
        plt.show()

        self.save_model('backup.model', backup['weights'])
        print(f'Saved backup model with score {backup["mse"]} from #{backup["epoch"]} epoch')

    def save_model(self, path, weights=None):
        if weights is None:
            weights = self.weights
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)