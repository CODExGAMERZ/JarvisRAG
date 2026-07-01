# Chapter 6: LLMs: Training, Alignment, and Quantization

This guide covers language model training pipelines from raw character slicing to human preference alignments and parameter-efficient adaptations.

---

## Part 1: Beginner: Text Generation and LLM Lifecycles

### 1. Autoregressive Text Generation
Large Language Models (LLMs) generate text by predicting the next token one-by-one:
- Input: "The sky is" -> Predicts: "blue"
- Next Input: "The sky is blue" -> Predicts: "and"
- Next Input: "The sky is blue and" -> Predicts: "clear"
Each predicted token is appended to the input context to generate the subsequent word.

### 2. Pre-Training Lifecycles
Models are pre-trained on massive datasets (like Web crawls, books) to learn grammar, vocabulary, and world facts by predicting hidden tokens in a self-supervised fashion.

---

## Part 2: Intermediate: Slicing Words and KV Caching

### 1. Tokenization: BPE vs. WordPiece
- **BPE (Byte-Pair Encoding)**: Groups frequent character pairs iteratively to build a vocabulary.
- **WordPiece**: Merges pairs that maximize the likelihood of the corpus:
  \[\text{Score}(A, B) = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}\]

### 2. Causal Masking
We apply an upper-triangular mask \(M\) to the self-attention logits to ensure tokens cannot attend to future positions:
\[M_{ij} = \begin{cases} 0 & i \ge j \\ -\infty & i < j \end{cases} \implies \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V\]

---

### 3. Key-Value (KV) Cache Python Implementation
Instead of recomputing key and value vectors for all previous tokens at each step, we cache them:

```python
import torch

class SimpleKVCache:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes: (batch_size, num_heads, sequence_length, head_dim)
        if self.k_cache is None:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            # Concatenate along the sequence length dimension (dim=2)
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
        return self.k_cache, self.v_cache
```

---

## Part 3: Expert: Human Alignment and DPO Proof

We align pre-trained models with human preferences using datasets \(\mathcal{D} = \{(x, y_w, y_l)\}\) where \(y_w\) is preferred over \(y_l\) for prompt \(x\).

---

### Mathematical Proof of the DPO Loss Objective
The RLHF objective is:
\[\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[ r(x, y) \right] - \beta \mathbb{D}_{\text{KL}}\left(\pi(y | x) \| \pi_{\text{ref}}(y | x)\right)\]
Using Lagrange multipliers, the optimal policy solution is:
\[\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left( \frac{r(x, y)}{\beta} \right)\]
where \(Z(x)\) is the partition function. Rearranging to solve for the reward \(r(x, y)\):
\[\ln \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} = \frac{r(x, y)}{\beta} - \ln Z(x) \implies r(x, y) = \beta \ln \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \ln Z(x)\]

The Bradley-Terry model defines the preference probability:
\[P(y_w \succ y_l | x) = \sigma\left( r(x, y_w) - r(x, y_l) \right)\]
Substituting our expression for \(r(x, y)\) into the Bradley-Terry model:
\[r(x, y_w) - r(x, y_l) = \left( \beta \ln \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} + \beta \ln Z(x) \right) - \left( \beta \ln \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)} + \beta \ln Z(x) \right)\]
The partition function term \(\beta \ln Z(x)\) cancels out:
\[r(x, y_w) - r(x, y_l) = \beta \ln \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \ln \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\]
Thus, we can write the preference probability solely in terms of policy likelihoods:
\[P(y_w \succ y_l | x) = \sigma\left( \beta \ln \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \ln \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)\]

Using Maximum Likelihood over the dataset, we obtain the DPO loss:
\[\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \ln \sigma \left( \beta \ln \frac{\pi_{\theta}(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \ln \frac{\pi_{\theta}(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]\]
This proof eliminates the need to train a reward model or use reinforcement learning.

---

## Part 4: OP Level: Quantized Adaptations (QLoRA)

### 1. Low-Rank Adaptation (LoRA)
Instead of updating full weight matrix \(W_0 \in \mathbb{R}^{d \times k}\), we update a low-rank decomposition:
\[W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A\]
where \(B \in \mathbb{R}^{d \times r}\) and \(A \in \mathbb{R}^{r \times k}\) with rank \(r \ll \min(d, k)\).

---

### 2. NormalFloat 4 (NF4) Quantization
NF4 is an optimal quantization type for zero-mean, unit-variance Gaussian distributions. The 16 NF4 quantization levels are:
```
[-1.0, -0.694, -0.515, -0.384, -0.275, -0.177, -0.083, 0.0,
 0.077, 0.160, 0.246, 0.339, 0.443, 0.570, 0.723, 1.0]
```
Base weights \(W^{\text{NF4}}\) are stored in 4-bit NF4. During the forward pass, they are dequantized to 16-bit precision and added to the LoRA updates:
\[Y = \text{Dequantize}\left(c_1, W^{\text{NF4}}\right) X + \frac{\alpha}{r} (X A B)\]
where \(c_1\) is the quantization constant scale factor.
