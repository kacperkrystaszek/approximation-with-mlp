import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class MultilayerPerceptron:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_layers = len(hidden_sizes) + 1
        self.activations = None
        self.layer_outputs = None

        # Initialize weights and biases for the network
        self.weights = [np.random.randn(input_size, hidden_sizes[0])]
        self.weights.extend([np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)])
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size))

        self.biases = [np.zeros((1, size)) for size in hidden_sizes]
        self.biases.append(np.zeros((1, output_size)))

    # Activation function
    def bisigmoid(self, x):
        return np.tanh(x)    

    def d_bisigmoid(self, x):
        return 1 - self.bisigmoid(x)**2

    def forward(self, x):
        self.activations = [x]
        self.layer_outputs = []

        for i in range(self.num_layers):
            layer_input = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.layer_outputs.append(layer_input)
            activation = self.bisigmoid(layer_input)
            self.activations.append(activation)

        return self.activations[-1]

    def backward(self, x, y, output):
        errors = [None] * self.num_layers
        deltas = [None] * self.num_layers

        errors[-1] = y - output
        deltas[-1] = errors[-1] * self.d_bisigmoid(output)

        for i in reversed(range(self.num_layers - 1)):
            errors[i] = deltas[i + 1].dot(self.weights[i + 1].T)
            deltas[i] = errors[i] * self.d_bisigmoid(self.activations[i + 1])

        for i in range(self.num_layers):
            if i == 0:
                self.weights[i] += x.T.dot(deltas[i]) * self.learning_rate
            else:
                self.weights[i] += self.activations[i].T.dot(deltas[i]) * self.learning_rate

            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * self.learning_rate

    def train(self, x, y, epochs):
        x_epochs = []
        mse_list = []
        
        for i in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)
            
            if not i % 500:
                pred = self.predict(x_test)
                mse = mean_squared_error(y_test, pred)
                x_epochs.append(i)
                mse_list.append(mse)
                print(f"Epoch {i}, MSE: {mse}")
            
        return x_epochs, mse_list

    def predict(self, x):
        return self.forward(x)

# Test data
x_test = np.array([float(x) for x in range(11)]).reshape(-1, 1)
y_test = np.array([1.0, 1.32, 1.6, 1.54, 1.41, 1.01, 0.6, 0.42, 0.2, 0.51, 0.8]).reshape(-1, 1)

# Create training data by recreating function
def create_func(x_values, y_values):
    result_x = []
    result_y = []
    
    for index, x in enumerate(x_values):
        if index + 1 != len(x_values):
            result_x.append(np.linspace(x-1, x, 11)[:-1])
        else:
            result_x.append(np.linspace(x-1, x, 11))
        
    for index, y in enumerate(y_values[1:], start=1):
        if index + 1 != len(y_values):
            result_y.append(np.linspace(y_values[index-1], y, 11)[:-1])
        else:
            result_y.append(np.linspace(y_values[index-1], y, 11))
    
    return np.concatenate(result_x).reshape(-1, 1), np.concatenate(result_y).reshape(-1, 1)

x, y = create_func(x_test[1:], y_test)

x_train = np.setdiff1d(x, x_test).reshape(-1, 1)
y_train = np.insert(np.setdiff1d(y, y_test, assume_unique=True), 58, 0.51).reshape(-1, 1) # There is duplicate of '0.51' value which has to be inserted manually

# Normalize the data between 0 and 1
y_max = y_train.max()
y_train /= y_max

# Create an instance of the MLP with 3 hidden layers with 100 neurons each
model = MultilayerPerceptron(input_size=1, hidden_sizes=[100, 100, 100], output_size=1, learning_rate=0.001)

# Train the model
x_epochs, mse_list = model.train(x_train, y_train, 10000)

# Test the model on new data
predictions = model.predict(x_test)

print(f"Final MSE: {mean_squared_error(y_test, predictions)}")

# MSE for every 500 epochs
plt.figure(figsize=(8, 6))
plt.plot(x_epochs, mse_list, color='b', label='MSE')
plt.title('MSE(epoch)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(x_train, y_train * y_max, color='b', label='Training data')
plt.plot(x_test, predictions * y_max, "r+", label='Predictions')
plt.title('Function approximation by multilayer perceptron with 3 hidden layers with 100 neurons each')
plt.xlabel('t[h]')
plt.ylabel('h(t)[m]')
plt.ylim(bottom=0, top=2)
plt.legend()
plt.show()
