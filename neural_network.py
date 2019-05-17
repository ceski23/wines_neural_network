import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

from nguyen_widrow import initnw


class NeuralNetwork:
    def __init__(self, lr, epochs, layers, err_ratio, lr_inc, lr_dec, goal):
        self.lr = lr
        self.epochs = epochs
        self.layers = layers
        self.err_ratio = err_ratio
        self.lr_inc = lr_inc
        self.lr_dec = lr_dec
        self.goal = goal
        self.weights = None
    
    def feed_training_data(self, P, T):
        self.P = P
        self.T = T
    
    def feed_test_data(self, P, T):
        self.test_P = P
        self.test_T = T

    def activation(self, x, derivative=False):
        return np.tanh(x) if not derivative else 1 - np.tanh(x)**2
    
    def linear(self, x, derivative=False):
        return 1 if derivative else x

    def init_weights_and_biases(self):
        self.weights, self.biases = [], []
        sizes = [self.P.shape[1], *self.layers, 1]

        for i in range(1, len(sizes)):
            self.weights.append(np.random.rand(sizes[i], sizes[i-1]))
            self.biases.append(np.random.rand(sizes[i]))

    # def init_weights_and_biases(self):
    #     self.weights, self.biases = [], []
    #     sizes = [self.P.shape[1], *self.layers]

    #     for i in range(1, len(sizes)):
    #         w, b = initnw(sizes[i], sizes[i-1])
    #         self.weights.append(w)
    #         self.biases.append(b)

    #     self.weights.append(np.random.rand(1, sizes[-1]))
    #     self.biases.append(np.random.rand(1))

    def predict(self, x):
        return self.forward(x)[0][-1][0]

    def forward(self, x):
        y, sum_inputs = [x], []
        for i in range(len(self.layers) + 1):
            f = self.linear if i == len(self.layers) else self.activation
            s = [np.dot(y[i], w) for w in self.weights[i]] + self.biases[i]      # [sum(y[i] * w) for w in self.weights[i]] + self.biases[i]
            y.append(f(s))
            sum_inputs.append(s)
        return y, sum_inputs

    def errors(self, d, y, sum_inputs):
        delta = [d - y[-1]]
        for k in range(len(self.layers), 0, -1):
            epsilon = [np.dot(delta[0], w) for w in self.weights[k].T]
            delta.insert(0, np.array(epsilon * self.activation(sum_inputs[k-1], True)))
        return delta
    
    def test(self, P, T):
        prediction = [self.predict(x) for x in P]
        error = np.array([d - y for y, d in zip(prediction, T)])
        cost = (error**2).sum()
        pk = int((np.abs(error) < 0.25).sum() / len(error) * 100)
        return prediction, cost, pk

    def update_weights_and_biases(self, delta, x):
        for i in range(len(self.layers) + 1):
            for j in range(len(self.weights[i])):
                factor = 2 * self.lr * delta[i][j]
                self.weights[i][j] += (factor * x[i])
                self.biases[i][j] += (factor * 1)
  
    def save_model(self, prefix='', weights=None):
        name = f'{self.costs[-1]:.5f}_{"_".join(map(str, self.layers))}'.replace(".", "")
        name = f'{prefix}_{name}' if len(prefix) > 0 else name
        weights = self.weights if weights is None else weights
        
        with open(f'models/{name}.mdl', 'wb') as f:
            pickle.dump((self.weights, self.layers, self.biases, self.lr), f)
            print(f'Zapisano model jako: {name}.mdl')
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            self.weights, self.layers, self.biases, self.lr = pickle.load(f)
    
    def update_learning_rate(self, cost):
        if len(self.costs) > 1:
            if cost > self.costs[-1] * self.err_ratio:
                self.lr = max(1e-10, self.lr * self.lr_dec)
                self.weights = self.last_weights
                self.biases = self.last_biases
                return False
            elif cost < self.costs[-1]:
                self.lr = min(1 - 1e-10, self.lr * self.lr_inc)
        return True

    def start_learning(self, live_plot=False, plot_interval=1, plot_results=False, live_text=False):
        if live_plot:
            plt.plot(self.test_T)
            plt.grid(linestyle='--')
            axes = plt.gca()
            axes.set_yticks(np.arange(min(self.test_T), max(self.test_T), 0.25), minor=True)
            line, = axes.plot([], [], 'r-')
            line.set_xdata(range(self.test_T.shape[0]))

        if self.weights is None:
            self.init_weights_and_biases()

        self.costs, self.pks = [], []
        self.last_weights, self.last_biases = [], []

        for epoch in range(self.epochs):
            for x, d in zip(self.P, self.T):
                y, sum_inputs = self.forward(x)
                delta = self.errors(d, y, sum_inputs)
                self.update_weights_and_biases(delta, y)

            prediction, cost, pk = self.test(self.test_P, self.test_T)

            if self.update_learning_rate(cost):
                self.costs.append(cost)
                self.pks.append(pk)
            else:
                self.costs.append(self.costs[-1])
                self.pks.append(self.pks[-1])

            if cost <= self.goal:
                print(f'\nAchieved goal with cost: {cost} after {epoch} epochs')
                break

            if pk == 100:
                print(f'\nAchieved 100%PK after {epoch} epochs')
                break

            self.last_weights = self.weights
            self.last_biases = self.biases

            if live_plot and not epoch % plot_interval:
                line.set_ydata(prediction)
                axes.set_title(f'Epoka #{epoch}\nCost: {cost:2.10f} \n LR: {self.lr}')
                plt.draw()
                plt.pause(1e-20)
            
            if live_text:
                print(f'Epoka #{epoch:05d}  Cost: {cost:14.10f}  LR: {self.lr:14.10f}  PK: {pk:3d}%', end='\r')
        
        if plot_results:
            plt.plot(prediction)
            plt.plot(self.test_T)

            plt.figure()
            plt.title('Cost')
            plt.plot(self.costs)

            plt.figure()
            plt.title('% PK')
            plt.plot(self.pks)

            with open('costs.txt', 'w') as f:
                for i, c in enumerate(self.costs):
                    f.write(f'{i} {c}\n')