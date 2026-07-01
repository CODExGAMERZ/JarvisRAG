# Chapter 4: Deep Learning Internals: Backprop, Initializations, and Norms

This guide covers neural network layers, activation functions, backpropagation derivations, initialization heuristics, and normalization implementations.

---

## Part 1: Beginner: Neurons and Multi-Layer Perceptrons (MLP)

### 1. What is an Artificial Neuron?
An artificial neuron is the building block of neural networks:
- **Inputs (\(x_i\))**: Features fed into the neuron.
- **Weights (\(w_i\))**: The importance/strength of each connection.
- **Bias (\(b\))**: An offset parameter added to the sum.
- **Activation Function (\(g\))**: A non-linear filter applied to the output.

The calculation is:
\[z = \sum_{i=1}^n w_i x_i + b \implies y = g(z)\]

### 2. Multi-Layer Perceptrons (MLPs)
An MLP consists of an input layer, one or more hidden layers, and an output layer.
For layer \(l \in \{1, \dots, L\}\):
\[Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}\]
\[A^{[l]} = g^{[l]}(Z^{[l]})\]
where \(W^{[l]}\) is the weight matrix of shape \((n^{[l]}, n^{[l-1]})\), \(b^{[l]}\) is the bias vector of shape \((n^{[l]}, 1)\), and \(A^{[0]} = X\) is the input.

---

## Part 2: Intermediate: Activations and Backpropagation Derivation

### 1. Common Activation Functions
- **ReLU (Rectified Linear Unit)**:
  \[f(z) = \max(0, z)\]
- **GELU (Gaussian Error Linear Unit)**:
  \[\text{GELU}(z) = z \Phi(z) \approx 0.5z \left( 1 + \tanh\left(\sqrt{\frac{2}{\pi}} (z + 0.044715z^3)\right) \right)\]
- **SwiGLU (Swish Gated Linear Unit)**:
  \[\text{SwiGLU}(x) = \left( x W_1 \odot \text{Swish}(x W_2) \right) W_3\]

---

### 2. Mathematical Derivation of Backpropagation
We compute the gradient of a scalar loss function \(\mathcal{L}\) with respect to every parameter.
Let the layer error term be:
\[\delta^{[l]} \equiv \frac{\partial \mathcal{L}}{\partial Z^{[l]}} \in \mathbb{R}^{n^{[l]} \times 1}\]

#### Output Layer Gradient (\(l = L\))
Using the chain rule:
\[\delta^{[L]}_i = \frac{\partial \mathcal{L}}{\partial Z^{[L]}_i} = \frac{\partial \mathcal{L}}{\partial A^{[L]}_i} \frac{\partial A^{[L]}_i}{\partial Z^{[L]}_i} = \frac{\partial \mathcal{L}}{\partial A^{[L]}_i} g^{[L]\prime}(Z^{[L]}_i)\]
In vector notation:
\[\delta^{[L]} = \nabla_{A^{[L]}} \mathcal{L} \odot g^{[L]\prime}(Z^{[L]})\]

#### Hidden Layer Gradients (\(l < L\))
To express \(\delta^{[l]}\) in terms of \(\delta^{[l+1]}\):
\[\delta^{[l]}_j = \frac{\partial \mathcal{L}}{\partial Z^{[l]}_j} = \sum_{k=1}^{n^{[l+1]}} \frac{\partial \mathcal{L}}{\partial Z^{[l+1]}_k} \frac{\partial Z^{[l+1]}_k}{\partial Z^{[l]}_j}\]
Since \(Z^{[l+1]}_k = \sum_{p=1}^{n^{[l]}} W^{[l+1]}_{kp} A^{[l]}_p + b^{[l+1]}_k\) and \(A^{[l]}_p = g^{[l]}(Z^{[l]}_p)\):
\[\frac{\partial Z^{[l+1]}_k}{\partial Z^{[l]}_j} = W^{[l+1]}_{kj} g^{[l]\prime}(Z^{[l]}_j)\]
Substituting this back:
\[\delta^{[l]}_j = \sum_{k=1}^{n^{[l+1]}} \delta^{[l+1]}_k W^{[l+1]}_{kj} g^{[l]\prime}(Z^{[l]}_j)\]
In vector notation:
\[\delta^{[l]} = \left( W^{[l+1]T} \delta^{[l+1]} \right) \odot g^{[l]\prime}(Z^{[l]})\]

#### Parameter Gradients
Now we calculate the derivatives with respect to the weights \(W^{[l]}\) and biases \(b^{[l]}\):
\[\frac{\partial \mathcal{L}}{\partial W^{[l]}_{ji}} = \frac{\partial \mathcal{L}}{\partial Z^{[l]}_j} \frac{\partial Z^{[l]}_j}{\partial W^{[l]}_{ji}} = \delta^{[l]}_j A^{[l-1]}_i \implies \frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (A^{[l-1]})^T\]
\[\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}\]

---

## Part 3: Expert: Weight Initialization Proofs

If weights are initialized too large, activations explode. If too small, they vanish. We derive weight initializations by tracking activation variances across layers.

Let \(z = \sum_{i=1}^{n_{\text{in}}} w_i x_i\). Assume weights and inputs are independent, mean-zero, and identically distributed:
\[\text{Var}(z) = n_{\text{in}} \text{Var}(w_i x_i) = n_{\text{in}} \left[ E(w_i)^2 \text{Var}(x_i) + E(x_i)^2 \text{Var}(w_i) + \text{Var}(w_i)\text{Var}(x_i) \right]\]
Since \(E(w_i) = 0\) and \(E(x_i) = 0\):
\[\text{Var}(z) = n_{\text{in}} \text{Var}(w) \text{Var}(x)\]

To maintain stable variance across layers (\(\text{Var}(z) = \text{Var}(x)\)):
- **Xavier (Glorot) Initialization (for Tanh/Sigmoid)**:
  Assumes linear activations:
  \[\text{Var}(w) = \frac{1}{n_{\text{in}}} \implies \text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}} \quad \text{(average bounds)}\]
- **He (Kaiming) Initialization (for ReLU)**:
  Since ReLU zero-outs negative activations, it halves the variance. To compensate, the variance of weights must be doubled:
  \[\text{Var}(w) = \frac{2}{n_{\text{in}}}\]

---

## Part 4: OP Level: Normalizations and PyTorch RMSNorm

Normalization stabilizes activation distributions across deep layers.

- **Batch Normalization**: Normalizes activations across the mini-batch dimension.
- **Layer Normalization**: Normalizes activations across the channel/feature dimension for a single token:
  \[\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2 \implies y = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\]
- **RMSNorm (Root Mean Square Normalization)**: Normalizes only by the root mean square, skipping mean subtraction to reduce computation:
  \[\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon} \implies y = \gamma \odot \frac{x}{\text{RMS}(x)}\]

### PyTorch Implementation of RMSNorm
Modern models (like LLaMA and Gemma) use RMSNorm for execution efficiency:

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean of squares along the last dimension
        variance = x.pow(2).mean(-1, keepdim=True)
        # Apply normalization and scale by gamma
        return x * torch.rsqrt(variance + self.eps) * self.gamma
```
