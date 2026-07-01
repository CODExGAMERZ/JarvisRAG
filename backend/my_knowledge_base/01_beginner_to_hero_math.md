# Chapter 1: Mathematics for AI/ML: From First Principles to Graduate Level

This guide is the complete math roadmap for AI/ML engineering, starting from absolute beginner basics and climbing to advanced research-level proofs.

---

## Part 1: Beginner Foundations: Coordinates, Vectors, and Matrices

### 1. Variables and the Coordinate Plane
A **variable** is a placeholder symbol (like \(x\) or \(y\)) representing a number that can change.
To visualize variables, we use a **Coordinate Plane (Cartesian Grid)**:
- The horizontal axis is the **X-axis**.
- The vertical axis is the **Y-axis**.
- Any point on the plane is written as \((x, y)\), representing steps taken from the origin \((0, 0)\).

#### Equations of Lines
A line represents a linear relationship between variables:
\[y = mx + c\]
where:
- \(m\): The **slope** (how steep the line is, calculated as \(\frac{\text{change in } y}{\text{change in } x}\)).
- \(c\): The **Y-intercept** (where the line crosses the vertical Y-axis).

---

### 2. Vectors: Arrows in Space
A **vector** is an ordered list of numbers. Geometrically, it represents an arrow pointing from the origin to a coordinate point.
Example of a 2D column vector:
\[v = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix}\]
- **Vector Addition**: Add corresponding elements:
  \[\begin{bmatrix} 1 \\ 3 \end{bmatrix} + \begin{bmatrix} 2 \\ -1 \end{bmatrix} = \begin{bmatrix} 1+2 \\ 3-1 \end{bmatrix} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}\]
- **Scalar Multiplication**: Scale the arrow's length by multiplying every element by a single number \(k\):
  \[k v = \begin{bmatrix} k v_1 \\ k v_2 \end{bmatrix}\]

---

### 3. Matrices: grids of Numbers
A **matrix** is a 2D grid of numbers with \(m\) rows and \(n\) columns (an \(m \times n\) matrix).
Example of a \(2 \times 3\) matrix:
\[A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}\]

#### Matrix Multiplication
To multiply matrix \(A \in \mathbb{R}^{m \times p}\) by matrix \(B \in \mathbb{R}^{p \times n}\), the number of columns in \(A\) must equal the number of rows in \(B\).
Let's calculate the element at row \(i\), column \(j\) of the resulting matrix \(C = AB\):
\[C_{ij} = \sum_{k=1}^p A_{ik} B_{kj}\]
Step-by-step example:
\[\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} (1\times 5 + 2\times 7) & (1\times 6 + 2\times 8) \\ (3\times 5 + 4\times 7) & (3\times 6 + 4\times 8) \end{pmatrix} = \begin{pmatrix} 19 & 22 \\ 43 & 50 \end{pmatrix}\]

---

## Part 2: Intermediate: Calculus, Slopes, and Gradients

### 1. Derivatives: Rates of Change
A **derivative** measures how fast a function's output changes when the input changes slightly. It is the slope of the tangent line at any point on a curve.

#### Limit Definition of a Derivative
For a function \(f(x)\):
\[f'(x) = \frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}\]

#### Common Derivatives
- Power Rule: \(\frac{d}{dx}(x^n) = n x^{n-1}\)
- Exponential Rule: \(\frac{d}{dx}(e^x) = e^x\)
- Chain Rule (for nested functions \(f(g(x))\)): \(\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}\)

---

### 2. Partial Derivatives and Gradients
When a function depends on multiple inputs, a **partial derivative** measures the rate of change with respect to one variable while holding all others constant. We use the symbol \(\partial\) instead of \(d\).
For \(f(x, y) = 3x^2 + 2y^3\):
- \(\frac{\partial f}{\partial x} = 6x\)
- \(\frac{\partial f}{\partial y} = 6y^2\)

The **Gradient** \(\nabla f\) gathers all partial derivatives into a vector, pointing in the direction of the steepest ascent:
\[\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}\]

---

## Part 3: Expert: Spectral Decompositions and Curvature

### 1. Eigenvalues and Eigenvectors
For a square matrix \(A \in \mathbb{R}^{n \times n}\), a non-zero vector \(v\) is an eigenvector and \(\lambda\) is its eigenvalue if:
\[A v = \lambda v\]
We find eigenvalues by solving \(\det(A - \lambda I) = 0\), and eigenvectors by finding the null space \(\text{Null}(A - \lambda I)\).

---

### 2. Singular Value Decomposition (SVD)
Any real matrix \(A \in \mathbb{R}^{m \times n}\) can be factored into:
\[A = U \Sigma V^T\]
where:
- \(U \in \mathbb{R}^{m \times m}\) is an orthogonal matrix (\(U^T U = I\)) of eigenvectors of \(A A^T\).
- \(\Sigma \in \mathbb{R}^{m \times n}\) is a diagonal matrix of singular values \(\sigma_i\) (where \(\sigma_i = \sqrt{\lambda_i}\) of \(A^T A\)).
- \(V \in \mathbb{R}^{n \times n}\) is an orthogonal matrix (\(V^T V = I\)) of eigenvectors of \(A^T A\).

#### Derivation of Right-Singular Vectors
Let's show that SVD aligns with the eigenvalues of the symmetric matrix \(A^T A\):
\[A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T\]
Since \(U^T U = I\):
\[A^T A = V (\Sigma^T \Sigma) V^T\]
Multiplying both sides by \(V\) on the right:
\[(A^T A) V = V (\Sigma^T \Sigma)\]
This proves that the columns of \(V\) are the eigenvectors of the symmetric matrix \(A^T A\), and the singular values are the square roots of the corresponding eigenvalues.

---

### 3. Jacobians and Hessians
- **Jacobian Matrix**: The matrix of all first-order partial derivatives for a vector-valued function \(f: \mathbb{R}^n \to \mathbb{R}^m\):
  \[J_{ij} = \frac{\partial f_i}{\partial x_j}\]
- **Hessian Matrix**: The square matrix of second-order partial derivatives for a scalar function \(f: \mathbb{R}^n \to \mathbb{R}\), measuring local curvature:
  \[H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}\]

---

## Part 4: OP Level: Advanced Probability and MAP Proof

### 1. Bayes' Theorem
Given conditional dependencies, we update the probability of a hypothesis \(H\) given evidence \(E\):
\[P(H|E) = \frac{P(E|H) P(H)}{P(E)}\]

---

### 2. Maximum Likelihood Estimation (MLE)
Given i.i.d. data points \(x_i\), we maximize the probability of the dataset:
\[\theta_{\text{MLE}} = \arg\max_{\theta} \sum_{i=1}^N \ln P(x_i | \theta)\]
For a Gaussian distribution \(\mathcal{N}(\mu, \sigma^2)\), taking the derivative of the log-likelihood with respect to \(\mu\) and setting it to zero yields the sample mean:
\[\mu_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N x_i\]

---

### 3. Mathematical Proof: Equivalence of MAP and L2 Regularization
Let the target variables follow a Gaussian distribution centered at the model prediction:
\[y_i \sim \mathcal{N}(w^T x_i, \sigma^2) \implies P(y_i | x_i, w) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - w^T x_i)^2}{2\sigma^2} \right)\]
Assume the weights prior follows a zero-mean Gaussian distribution with variance \(\sigma_0^2\):
\[w_j \sim \mathcal{N}(0, \sigma_0^2) \implies P(w_j) = \frac{1}{\sqrt{2\pi\sigma_0^2}} \exp\left( -\frac{w_j^2}{2\sigma_0^2} \right)\]

The Maximum A Posteriori (MAP) estimation objective is:
\[w_{\text{MAP}} = \arg\max_{w} P(Y | X, w) P(w)\]
Taking the natural logarithm:
\[w_{\text{MAP}} = \arg\max_{w} \sum_{i=1}^N \ln P(y_i | x_i, w) + \sum_{j=1}^D \ln P(w_j)\]
Substituting the Gaussian densities:
\[w_{\text{MAP}} = \arg\max_{w} \sum_{i=1}^N \left[ -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(y_i - w^T x_i)^2}{2\sigma^2} \right] + \sum_{j=1}^D \left[ -\frac{1}{2}\ln(2\pi\sigma_0^2) - \frac{w_j^2}{2\sigma_0^2} \right]\]
Discarding constant terms that do not depend on \(w\):
\[w_{\text{MAP}} = \arg\max_{w} \sum_{i=1}^N -\frac{(y_i - w^T x_i)^2}{2\sigma^2} - \sum_{j=1}^D \frac{w_j^2}{2\sigma_0^2}\]
We convert this maximization to a minimization problem by multiplying by \(-2\sigma^2\):
\[w_{\text{MAP}} = \arg\min_{w} \sum_{i=1}^N (y_i - w^T x_i)^2 + \frac{\sigma^2}{\sigma_0^2} \sum_{j=1}^D w_j^2\]
Letting \(\lambda = \frac{\sigma^2}{\sigma_0^2}\) be the regularization parameter:
\[w_{\text{MAP}} = \arg\min_{w} \|Y - Xw\|_2^2 + \lambda \|w\|_2^2\]
This is exactly the L2 Regularization (Ridge Regression) objective function. This completes the proof that Ridge regression is mathematically equivalent to MAP parameter estimation with a Gaussian prior over the weights.
