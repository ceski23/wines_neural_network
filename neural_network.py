import matplotlib.pyplot as plt
import numpy as np
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
        return self.forward(x)[0][-1][0]
##################################################################################################################################
    def forward(self, x):
        '''Oblicza sygnał wyjściowy neuronów'''
        y, ss = [x], []
        for i in range(len(self.layers) + 1):
            f = self.linear if i == len(self.layers) else self.sigmoid
            s = np.array([sum(y[i] * w) for w in self.weights[i]])      # w - wagi jednego neuronu
            y.append(f(s))
            ss.append(s)
        return y, ss

    def errors(self, d, y, ss):
        delta = [d - y[-1]]
        for k in range(len(self.layers), 0, -1):
            epsilon = [sum(delta[0] * w) for w in self.weights[k].T]    # w - wagi jednego neuronu
            delta.insert(0, np.array(epsilon * self.sigmoid(ss[k-1], True)))
        return delta
    
    def test(self, P, T):
        prediction = [self.predict(x) for x in P]
        error = [d - y for y, d in zip(prediction, T)]
        cost = max([(e**2) for e in error])
        return prediction, cost

    def update_weights(self, w, d, x):
        '''Obliczanie wag dla pojedynczego neuronu'''
        return w + (2 * self.lr * d * x)
##################################################################################################################################
  
    def save_model(self, prefix='', weights=None):
        name = f'{self.costs[-1]:.5f}_{"_".join(map(str, self.layers))}'.replace(".", "")
        name = f'{prefix}_{name}' if len(prefix) > 0 else name
        weights = self.weights if weights is None else weights
        
        with open(f'models/{name}.mdl', 'wb') as f:
            pickle.dump((self.weights, self.layers), f)
            print(f'Zapisano model jako: {name}.mdl')
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.weights, self.layers = pickle.load(f)
    
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
##################################################################################################################################
        for epoch in range(self.epochs):
            for x, d in zip(self.P, self.T):
                y, ss = self.forward(x)
                delta = self.errors(d, y, ss)

                for i in range(len(self.layers) + 1):
                    for j in range(len(self.weights[i])):
                        self.weights[i][j] = self.update_weights(self.weights[i][j], delta[i][j], y[i])
##################################################################################################################################

            prediction, cost = self.test(self.test_P, self.test_T)
            self.costs.append(cost)
##################################################################################################################################

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