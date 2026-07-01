# Chapter 5: Attention and Transformers: Mechanics and Architectures

This guide covers attention mechanisms, positional encodings, and Transformer block layer organizations.

---

## Part 1: Beginner: Sequence Data and Attention Intuition

### 1. What is Sequence Data?
Sequence data is data where the order of items matters (e.g. text, time series, audio).
- **Bag of Words**: Simple method that counts word frequencies but ignores order entirely (e.g. "not bad" and "bad, not" look identical).
- **Recurrent Neural Networks (RNNs)**: Process text one word at a time, passing a hidden memory state \(h_t\) forward. However, they are slow to train because they cannot compute in parallel.

### 2. Attention Intuition
Instead of reading step-by-step, **Attention** allows the model to look at all words in a sequence simultaneously and weigh their relative importance.
- If you read: "The **bank** of the river", attention links "bank" strongly with "river".
- If you read: "The money is in the **bank**", attention links "bank" strongly with "money".

---

## Part 2: Intermediate: Scaled Dot-Product Attention

Given an input matrix \(X \in \mathbb{R}^{T \times d_{\text{model}}}\), we project it into Queries (\(Q\)), Keys (\(K\)), and Values (\(V\)) using learned weight matrices:
\[Q = X W_Q \in \mathbb{R}^{T \times d_k}, \quad K = X W_K \in \mathbb{R}^{T \times d_k}, \quad V = X W_V \in \mathbb{R}^{T \times d_v}\]
The attention equation is:
\[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V\]

---

### Mathematical Proof of the Scaling Factor \(\sqrt{d_k}\)
Let \(q \in \mathbb{R}^{d_k}\) and \(k \in \mathbb{R}^{d_k}\) represent a query vector and a key vector. Assume components \(q_i\) and \(k_i\) are independent random variables with:
\[E(q_i) = E(k_i) = 0 \quad \text{and} \quad \text{Var}(q_i) = \text{Var}(k_i) = 1\]
The dot product is:
\[q \cdot k = \sum_{j=1}^{d_k} q_j k_j\]
1. **Expected Value**:
   \[E(q \cdot k) = \sum_{j=1}^{d_k} E(q_j k_j) = \sum_{j=1}^{d_k} E(q_j) E(k_j) = 0\]
2. **Variance**: Since \(q_j\) and \(k_j\) are independent:
   \[\text{Var}(q_j k_j) = E(q_j^2 k_j^2) - [E(q_j k_j)]^2 = E(q_j^2) E(k_j^2) - 0\]
   Since \(E(q_j^2) = \text{Var}(q_j) + [E(q_j)]^2 = 1\):
   \[\text{Var}(q_j k_j) = 1 \times 1 = 1\]
   The variance of the sum of \(d_k\) independent variables is:
   \[\text{Var}(q \cdot k) = \sum_{j=1}^{d_k} \text{Var}(q_j k_j) = d_k\]

If the variance of the dot product is \(d_k\), the values can grow large in magnitude, pushing the softmax function into regions with extremely small gradients. To force the variance back to 1, we scale the dot product by \(\frac{1}{\sqrt{d_k}}\):
\[\text{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{1}{d_k} \text{Var}(q \cdot k) = \frac{d_k}{d_k} = 1\]

---

## Part 3: Expert: Multi-Head Attention and RoPE

### 1. Multi-Head Attention
We project Queries, Keys, and Values \(h\) times to attend to different representation subspaces:
\[\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O\]
where \(\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)\) with head dimension \(d_k = d_v = \frac{d_{\text{model}}}{h}\).

---

### 2. Rotary Position Embedding (RoPE)
RoPE represents 2D slices of the embedding vector in the complex plane, rotating them by an angle proportional to position index \(m\).
For a 2D vector \(x = [x_1, x_2]^T\):
\[R_{\Theta, m}^2 x = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}\]
For a \(d\)-dimensional vector \(x \in \mathbb{R}^d\), we partition it into \(d/2\) sub-vectors and apply rotations:
\[R_{\Theta, m}^d x = \text{diag}\left( R_{\Theta, m, 1}^2, \dots, R_{\Theta, m, d/2}^2 \right) x \quad \text{where} \quad \theta_i = 10000^{-2(i-1)/d}\]

#### Relative Distance Proof
Since \(R_{\Theta, m}^d\) is orthogonal (\((R_{\Theta, m}^d)^T R_{\Theta, m}^d = I\)), the dot product of a rotated query at \(m\) and key at \(n\) is:
\[(R_{\Theta, m}^d q)^T (R_{\Theta, n}^d k) = q^T (R_{\Theta, m}^d)^T R_{\Theta, n}^d k = q^T R_{\Theta, n-m}^d k\]
This proves that the attention score between two tokens depends only on their relative distance \(n-m\).

---

## Part 4: OP Level: Pre-LN vs. Post-LN Block Layouts

### 1. Post-LN Block Layout
Normalizes the representation after the residual connection:
\[x_{l+1} = \text{LN}\left(x_l + F_l(x_l)\right)\]
- **Problem**: The scale factor of the normalizer decreases gradients exponentially in early layers, requiring learning rate warmups.

### 2. Pre-LN Block Layout
Normalizes the input to each sub-layer before processing:
\[x_{l+1} = x_l + F_l\left(\text{LN}(x_l)\right)\]

#### Proof: Direct Gradient Flow
Let's show the gradient of the final state \(x_L\) with respect to an early state \(x_l\):
\[x_L = x_l + \sum_{k=l}^{L-1} F_k\left(\text{LN}(x_k)\right)\]
Differentiating both sides with respect to \(x_l\):
\[\frac{\partial x_L}{\partial x_l} = I + \sum_{k=l}^{L-1} \frac{\partial F_k\left(\text{LN}(x_k)\right)}{\partial x_l}\]
The identity matrix \(I\) guarantees that gradients propagate back directly without scaling, preventing vanishing gradients in deep architectures.
