import matplotlib.pyplot as plt
import numpy as np
from time import time
import pickle
import math


class NeuralNetwork:
    def __init__(self, lr, epochs, layers, err_ratio, lr_inc, lr_dec, goal):
        self.lr = lr
        self.epochs = epochs
        self.layers = layers
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

    def init_weights(self):
        self.weights = []
        sizes = [self.P.shape[1], *self.layers, 1]
        for i in range(1, len(sizes)):
            self.weights.append(np.random.rand(sizes[i], sizes[i-1]))

    def predict(self, x):
        y = [x]
        for i in range(len(self.weights)):
            f = self.linear if i == len(self.weights) - 1 else self.sigmoid
            s = np.array([self.forward(y[i], w) for w in self.weights[i]])
            y.append(f(s))
        return y[-1][0]

    def forward(self, x, w):
        '''Oblicza wyjście dla pojedynczego neuronu'''
        return sum(x * w)

    def neuron_error(self, l, w):
        '''Propagacja błędu w jednej warstwie'''
        return sum(l * w)

    def update_weights(self, x, w, func, l, e):
        '''Obliczanie wag dla pojedynczego neuronu'''
        d = self.lr * l * func(e, True) * x
        return w + d
    
  
    def save_model(self, prefix='', weights=None):
        name = f'{self.costs[-1]:.5f}_{"_".join(map(str, self.layers))}'.replace(".", "")
        name = f'{prefix}_{name}' if len(prefix) > 0 else name
        weights = self.weights if weights is None else weights
        
        with open(f'models/{name}.mdl', 'wb') as f:
            pickle.dump(self.weights, f)
            print(f'Zapisano model jako: {name}.mdl')
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)  
    
    def start_learning(self, live_plot=False, plot_interval=1):
        if live_plot:
            plt.plot(self.test_T)
            plt.grid(linestyle='--')
            axes = plt.gca()
            axes.set_yticks(np.arange(min(self.test_T), max(self.test_T), 0.25), minor=True)
            line, = axes.plot([], [], 'r-')
            line.set_xdata(range(self.test_T.shape[0]))

        self.init_weights()
        self.costs = []

        last_weights = []
        backup = {'weights': None, 'cost': math.inf, 'epoch': -1}

        for epoch in range(self.epochs):
            for x, d in zip(self.P, self.T):
                y, ss = [x], []

                for i in range(len(self.weights)):
                    f = self.linear if i == len(self.weights) - 1 else self.sigmoid
                    s = np.array([self.forward(y[i], w) for w in self.weights[i]])
                    y.append(f(s))
                    ss.append(s)

                errors = [d - y[-1]]
                for layer_idx in range(len(self.weights)-1, 0, -1):
                    layer_error = [self.neuron_error(errors[0], w) for w in self.weights[layer_idx].T]
                    errors.insert(0, layer_error)

                for i in range(len(self.weights)):
                    func = self.linear if i == len(self.weights) - 1 else self.sigmoid
                    for j in range(len(self.weights[i])):
                        self.weights[i][j] = self.update_weights(y[i], self.weights[i][j], func, errors[i][j], ss[i][j])


            prediction = [self.predict(x) for x in self.test_P]
            error = [d - y for y, d in zip(prediction, self.test_T)]
            cost = max([0.5 * (e**2) for e in error])
            self.costs.append(cost)

            if cost <= self.goal:
                print(f'Achieved goal with cost: {cost}')
                break

            if len(self.costs) >= 2:
                if self.costs[-1] > self.costs[-2] * self.err_ratio:
                    self.lr *= self.lr_dec
                    self.weights = last_weights
                elif self.costs[-1] < self.costs[-2]: #*(2 - self.err_ratio):
                    self.lr *= self.lr_inc

            last_weights = self.weights

            if cost < backup['cost']:
                backup['cost'] = cost
                backup['weights'] = self.weights
                backup['epoch'] = epoch

            if live_plot and not epoch % plot_interval:
                line.set_ydata(prediction)
                axes.set_title(f'Epoka #{epoch}\ncost: {cost:2.10f} \n LR: {self.lr}')
                plt.draw()
                plt.pause(1e-20)
            
            if not live_plot:
                print(f'Epoka #{epoch:02d} cost: {cost:2.10f}', end='\r')
        
        if not live_plot:
            plt.plot(prediction)
            plt.plot(self.test_T)

        plt.figure()
        plt.plot(self.costs)
        plt.show()

        if backup['cost'] != cost:
            self.save_model(f'BCP_{backup["epoch"]}_', backup['weights'])