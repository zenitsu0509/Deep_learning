import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Generate sample data
x = np.linspace(-10, 10, 400)

# Compute activation functions
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_softmax = softmax(np.vstack([x, x + 1, x - 1]))  # Softmax expects a 2D input

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Function')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, y_tanh, label='Tanh')
plt.title('Tanh Function')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, y_relu, label='ReLU')
plt.title('ReLU Function')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.title('Leaky ReLU Function')
plt.grid(True)

plt.subplot(2, 3, 5)
for i in range(y_softmax.shape[0]):
    plt.plot(x, y_softmax[i], label=f'Softmax curve {i+1}')
plt.title('Softmax Function')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
