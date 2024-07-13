# Activation Functions in Neural Networks

Activation functions play a crucial role in neural networks. They introduce non-linearity into the network, enabling it to learn and model complex data patterns. Without activation functions, a neural network would essentially behave like a linear regression model, regardless of its depth.

## Overview of Common Activation Functions

### Sigmoid Function

\[ \sigma(x) = \frac{1}{1 + e^{-x}} \]

- **Output range**: (0, 1)
- **Commonly used in**: Binary classification tasks.
- **Issues**: Can cause vanishing gradient problems during backpropagation.

### Tanh (Hyperbolic Tangent) Function

\[ \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \]

- **Output range**: (-1, 1)
- **Often preferred over sigmoid**: Its output is zero-centered.
- **Issues**: Can also cause vanishing gradient problems.

### ReLU (Rectified Linear Unit)

\[ \text{ReLU}(x) = \max(0, x) \]

- **Output range**: [0, ∞)
- **Widely used due to**: Its simplicity and effectiveness.
- **Issues**: Can cause "dying ReLU" where neurons output zero for all inputs.

### Leaky ReLU

\[ \text{Leaky ReLU}(x) = \begin{cases} 
x & \text{if } x > 0 \\ 
\alpha x & \text{otherwise} 
\end{cases} \]

- **Output range**: (-∞, ∞)
- **Helps to mitigate the "dying ReLU" problem**: By allowing a small, non-zero gradient when the input is negative.

### Softmax Function

\[ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} \]

- **Output range**: (0, 1), where all outputs sum to 1.
- **Commonly used in the output layer of a classifier**: To represent a probability distribution over classes.
