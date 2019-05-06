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
        f = 1.0 / (1.0 + np.exp(-x))
        return f if not derivative else f * (1 - f)
    
    def linear(self, x, derivative=False):
        return 1 if derivative else x

    def init_weights(self):
        self.weights = []
        sizes = [self.P.shape[1], *self.layers, 1]
        for i in range(1, len(sizes)):
            self.weights.append(np.random.rand(sizes[i], sizes[i-1]))
            # self.weights.append(np.ones((sizes[i], sizes[i-1])))

    def init_biases(self):
        self.biases = []
        layers = [*self.layers, 1]
        for layer in layers:
            self.biases.append(np.random.rand(layer))
            # self.biases.append(np.ones((layer)))

    def predict(self, x):
        return self.forward(x)[0][-1][0]

    def forward(self, x):
        y, sum_inputs = [x], []
        for i in range(len(self.layers) + 1):
            f = self.linear if i == len(self.layers) else self.sigmoid
            s = [np.dot(y[i], w) for w in self.weights[i]] + self.biases[i]      # [sum(y[i] * w) for w in self.weights[i]] + self.biases[i]
            y.append(f(s))
            sum_inputs.append(s)
        return y, sum_inputs

    def errors(self, d, y, sum_inputs):
        delta = [d - y[-1]]
        for k in range(len(self.layers), 0, -1):
            epsilon = [np.dot(delta[0], w) for w in self.weights[k].T]
            delta.insert(0, np.array(epsilon * self.sigmoid(sum_inputs[k-1], True)))
        return delta
    
    def test(self, P, T):
        prediction = [self.predict(x) for x in P]
        error = [d - y for y, d in zip(prediction, T)]
        cost = sum([(e**2) for e in error])
        return prediction, cost

    def update_weights_and_biases(self, delta, x):
        for i in range(len(self.layers) + 1):
            for j in range(len(self.weights[i])):
                self.weights[i][j] += (2 * self.lr * delta[i][j] * x[i])
                self.biases[i][j] += (self.lr * delta[i][j])
  
    def save_model(self, prefix='', weights=None):
        name = f'{self.costs[-1]:.5f}_{"_".join(map(str, self.layers))}'.replace(".", "")
        name = f'{prefix}_{name}' if len(prefix) > 0 else name
        weights = self.weights if weights is None else weights
        
        with open(f'models/{name}.mdl', 'wb') as f:
            pickle.dump((self.weights, self.layers, self.biases), f)
            print(f'Zapisano model jako: {name}.mdl')
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.weights, self.layers, self.biases = pickle.load(f)
    
    def start_learning(self, live_plot=False, plot_interval=1, plot_results=False, live_text=False):
        if live_plot:
            plt.plot(self.test_T)
            plt.grid(linestyle='--')
            axes = plt.gca()
            axes.set_yticks(np.arange(min(self.test_T), max(self.test_T), 0.25), minor=True)
            line, = axes.plot([], [], 'r-')
            line.set_xdata(range(self.test_T.shape[0]))

        self.init_weights()
        self.init_biases()
        self.costs, self.pks = [], []

        last_weights = []
        for epoch in range(self.epochs):
            for x, d in zip(self.P, self.T):
                y, sum_inputs = self.forward(x)
                delta = self.errors(d, y, sum_inputs)
                self.update_weights_and_biases(delta, y)

            prediction, cost = self.test(self.test_P, self.test_T)
            self.costs.append(cost)
            pk = [abs(x - y) < 0.25 for x, y in zip(prediction, self.test_T)]
            result = int((sum(pk) / len(pk)) * 100)
            self.pks.append(result)

            if cost <= self.goal:
                print(f'\nAchieved goal with cost: {cost} after {epoch} epochs')
                break

            if result == 100:
                print(f'\nAchieved 100%PK after {epoch} epochs')
                break

            if len(self.costs) >= 2:
                if self.costs[-1] > self.costs[-2] * self.err_ratio:
                    self.lr = max(1e-10, self.lr * self.lr_dec)
                    self.weights = last_weights
                elif self.costs[-1] < self.costs[-2]: #*(2 - self.err_ratio):
                    self.lr = min(1 - 1e-10, self.lr * self.lr_inc)

            last_weights = self.weights

            if live_plot and not epoch % plot_interval:
                line.set_ydata(prediction)
                axes.set_title(f'Epoka #{epoch}\nCost: {cost:2.10f} \n LR: {self.lr}')
                plt.draw()
                plt.pause(1e-20)
            
            if live_text:
                print(f'Epoka #{epoch:05d}  Cost: {cost:14.10f}  LR: {self.lr:14.10f}  PK: {result:3d}%', end='\r')
        
        if plot_results:
            plt.plot(prediction)
            plt.plot(self.test_T)

            plt.figure()
            plt.plot(self.costs)

            plt.figure()
            plt.plot(self.pks)