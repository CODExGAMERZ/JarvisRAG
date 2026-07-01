# Chapter 3: Classical Machine Learning: From Scratch to Optimization

This guide covers classical machine learning algorithms, their mathematical optimization steps, and gradient updates.

---

## Part 1: Beginner: Data, Models, and Generalization

### 1. What is Data? Features and Labels
Machine learning models learn patterns from data:
- **Features (\(X\))**: The input variables used to make predictions (e.g. size of a house, number of bedrooms).
- **Labels (\(y\))**: The target output variable we want to predict (e.g. price of the house).

### 2. Supervised vs. Unsupervised Learning
- **Supervised Learning**: The dataset contains both features and labels. The model is guided by feedback (loss) to match the targets.
- **Unsupervised Learning**: The dataset contains features but no labels. The model identifies structure on its own (e.g. clustering).

### 3. Training, Validation, and Test Splits
To evaluate models fairly:
- **Training Set**: Used to adjust the model weights.
- **Validation Set**: Used to tune hyperparameters (e.g. learning rate) and prevent overfitting.
- **Test Set**: Kept completely hidden until final evaluation to measure generalization.

---

### 4. Overfitting vs. Underfitting
- **Underfitting (High Bias)**: The model is too simple. It cannot fit the training data.
- **Overfitting (High Variance)**: The model memorizes noise in the training data, leading to poor performance on new data.

---

## Part 2: Intermediate: Linear and Logistic Regressions

### 1. Linear Regression and the Normal Equations
Linear regression maps outputs as a linear combination of inputs:
\[\hat{y} = w^T x\]
Given design matrix \(X \in \mathbb{R}^{N \times D}\) and target vector \(Y \in \mathbb{R}^N\), we minimize the Mean Squared Error (MSE) loss:
\[J(w) = \frac{1}{2N} (Y - Xw)^T (Y - Xw) = \frac{1}{2N} \left( Y^T Y - 2 w^T X^T Y + w^T X^T X w \right)\]

#### Closed-Form Analytical Derivation
To find the optimal weight vector \(w^*\), we compute the derivative of the cost function with respect to \(w\) and set it to zero:
\[\nabla_w J(w) = \frac{1}{N} \left( -X^T Y + X^T X w \right) = 0\]
\[X^T X w = X^T Y\]
Multiplying both sides by the inverse \((X^T X)^{-1}\) (assuming the matrix is non-singular):
\[w^* = (X^T X)^{-1} X^T Y\]

---

### 2. Logistic Regression and BCE Gradient Derivation
For binary classification, we squish the linear output \(z_i = w^T x_i\) between 0 and 1 using the Sigmoid function:
\[p(y_i=1 | x_i; w) = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}\]
The Binary Cross-Entropy (BCE) cost function is derived from the negative log-likelihood:
\[J(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(z_i) + (1-y_i) \ln(1 - \sigma(z_i)) \right]\]

#### Complete Step-by-Step Gradient Derivation
Let \(\hat{y}_i = \sigma(z_i)\) where \(z_i = w^T x_i\). We differentiate \(J(w)\) with respect to weight parameter \(w_j\) using the chain rule:
\[\frac{\partial J(w)}{\partial w_j} = \sum_{i=1}^N \frac{\partial J}{\partial \hat{y}_i} \frac{\partial \hat{y}_i}{\partial z_i} \frac{\partial z_i}{\partial w_j}\]

1. **First Term**: Differentiating the cost function with respect to prediction \(\hat{y}_i\):
   \[\frac{\partial J}{\partial \hat{y}_i} = -\left( \frac{y_i}{\hat{y}_i} - \frac{1 - y_i}{1 - \hat{y}_i} \right) = \frac{\hat{y}_i - y_i}{\hat{y}_i(1 - \hat{y}_i)}\]
2. **Second Term**: Differentiating the Sigmoid prediction with respect to logit \(z_i\):
   \[\frac{\partial \hat{y}_i}{\partial z_i} = \frac{d}{dz_i} (1 + e^{-z_i})^{-1} = -(1 + e^{-z_i})^{-2}(-e^{-z_i}) = \frac{1}{1 + e^{-z_i}} \frac{e^{-z_i}}{1 + e^{-z_i}} = \hat{y}_i (1 - \hat{y}_i)\]
3. **Third Term**: Differentiating the logit with respect to weight \(w_j\):
   \[\frac{\partial z_i}{\partial w_j} = \frac{\partial}{\partial w_j} \left( w_0 x_{i0} + \dots + w_j x_{ij} + \dots + w_D x_{iD} \right) = x_{ij}\]

Multiplying these three terms together:
\[\frac{\partial J(w)}{\partial w_j} = \sum_{i=1}^N \left[ \frac{\hat{y}_i - y_i}{\hat{y}_i(1 - \hat{y}_i)} \right] \cdot \left[ \hat{y}_i(1 - \hat{y}_i) \right] \cdot x_{ij}\]
Notice that \(\hat{y}_i(1 - \hat{y}_i)\) cancels out:
\[\frac{\partial J(w)}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) x_{ij}\]
In vector form:
\[\nabla_w J(w) = \frac{1}{N} X^T (\hat{Y} - Y)\]

---

## Part 3: Expert: Regularization Mechanics

Regularization penalizes weight magnitudes to constrain model capacity and prevent overfitting.

- **L2 Regularization (Ridge)**: Adds a squared penalty:
  \[J(w) = J_0(w) + \frac{\lambda}{2} \sum_{j=1}^D w_j^2\]
  The boundary constraint is a hypersphere (circle in 2D). It shrinks all weights towards zero smoothly.
- **L1 Regularization (Lasso)**: Adds an absolute penalty:
  \[J(w) = J_0(w) + \lambda \sum_{j=1}^D |w_j|\]
  The boundary constraint is a hyperoctahedron (diamond in 2D). Because the corners of the diamond lie on the coordinate axes, optimization updates frequently hit these corners, forcing some weight coordinates to exactly zero.

---

## Part 4: OP Level: Momentum and Adam Optimizers

Optimizers search the parameter space to locate the minimum of the loss surface.

### 1. SGD with Momentum
Momentum uses a moving average of previous gradients to accelerate updates along directions of consistent descent, dampening oscillations:
\[v_t = \beta v_{t-1} + (1 - \beta) \nabla_{\theta} J(\theta_t)\]
\[\theta_{t+1} = \theta_t - \alpha v_t\]
where \(\beta \in [0, 1)\) is the momentum decay factor.

### 2. Adam (Adaptive Moment Estimation)
Adam calculates adaptive learning rates for each parameter by tracking the first and second moments:
1. **First Moment (Gradient Mean)**:
   \[m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\]
2. **Second Moment (Uncentered Variance)**:
   \[v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\]
   where \(g_t = \nabla_{\theta} J(\theta_t)\).

3. **Bias Correction**: Since \(m_t\) and \(v_t\) are initialized to zero, they are biased towards zero at the start of training. We correct this bias:
   \[\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}\]
4. **Parameter Update**:
   \[\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t\]
   where \(\alpha\) is the learning rate, and \(\epsilon\) prevents division by zero.
