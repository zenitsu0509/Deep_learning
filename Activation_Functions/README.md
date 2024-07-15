# Activation Functions in Neural Networks

Activation functions play a crucial role in neural networks. They introduce non-linearity into the network, enabling it to learn and model complex data patterns. Without activation functions, a neural network would essentially behave like a linear regression model, regardless of its depth.

## Overview of Common Activation Functions

### Sigmoid Function

<p align="center">
  &sigma;(x) = &#49; / (&#49; + e<sup>-x</sup>)
</p>

- **Output range**: (0, 1)
- **Commonly used in**: Binary classification tasks.
- **Issues**: Can cause vanishing gradient problems during backpropagation.

### Tanh (Hyperbolic Tangent) Function

<p align="center">
  tanh(x) = (e<sup>x</sup> - e<sup>-x</sup>) / (e<sup>x</sup> + e<sup>-x</sup>)
</p>

- **Output range**: (-1, 1)
- **Often preferred over sigmoid**: Its output is zero-centered.
- **Issues**: Can also cause vanishing gradient problems.

### ReLU (Rectified Linear Unit)

<p align="center">
  ReLU(x) = max(0, x)
</p>

- **Output range**: [0, ∞)
- **Widely used due to**: Its simplicity and effectiveness.
- **Issues**: Can cause "dying ReLU" where neurons output zero for all inputs.

### Leaky ReLU

<p align="center">
  Leaky ReLU(x) = { x if x &gt; 0 <br> &alpha;x otherwise }
</p>

- **Output range**: (-∞, ∞)
- **Helps to mitigate the "dying ReLU" problem**: By allowing a small, non-zero gradient when the input is negative.

### Softmax Function

<p align="center">
  softmax(x<sub>i</sub>) = e<sup>x<sub>i</sub></sup> / &sum;<sub>j</sub> e<sup>x<sub>j</sub></sup>
</p>

- **Output range**: (0, 1), where all outputs sum to 1.
- **Commonly used in the output layer of a classifier**: To represent a probability distribution over classes.
