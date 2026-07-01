# Chapter 2: Computer Systems and Programming (Python, C++, CUDA)

This guide takes you from basic programming structures to GPU architecture and parallel CUDA kernels.

---

## Part 1: Beginner: Basic Programming Structures

### 1. Variables and Data Types
A **variable** is a labeled box in your computer's memory that stores information.
In Python, variables are created when you assign them values using the `=` operator:
```python
# Integer (whole number)
age = 25

# Float (decimal number)
learning_rate = 0.01

# String (text)
model_name = "Gemini"

# Boolean (true or false)
is_trained = False
```

### 2. Control Flow: Decisions and Loops
- **Conditionals (`if`, `elif`, `else`)**: Executing code selectively:
  ```python
  if score > 90:
      print("Excellent!")
  elif score > 70:
      print("Pass")
  else:
      print("Fail")
  ```
- **Loops (`for` and `while`)**: Repeating tasks:
  ```python
  # Iterate over a range of numbers (0, 1, 2, 3, 4)
  for i in range(5):
      print(f"Step {i}")
  ```

### 3. Data Structures
- **List**: An ordered, mutable collection of items:
  ```python
  fruits = ["apple", "banana", "cherry"]
  fruits.append("orange")
  print(fruits[0]) # Output: apple
  ```
- **Dictionary**: Key-value pairs:
  ```python
  model_config = {
      "batch_size": 32,
      "epochs": 10,
      "optimizer": "Adam"
  }
  print(model_config["batch_size"]) # Output: 32
  ```

---

## Part 2: Intermediate: OOP and NumPy Layouts

### 1. Object-Oriented Programming (OOP)
OOP organizes code into reusable blueprints called **Classes**, which create specific instances called **Objects**.
```python
class Model:
    def __init__(self, name: str, params: int):
        self.name = name
        self.params = params  # Attribute
        
    def summary(self):      # Method
        print(f"Model: {self.name} | Parameters: {self.params}")

# Instantiate an object
my_model = Model("AntigravityRAG", 1200000)
my_model.summary()
```

### 2. Python Memory Management
CPython tracks memory references using an internal header `ob_refcnt`. When an object's reference count drops to 0, its memory is immediately reclaimed.

#### Cyclic Garbage Collection
To catch isolated reference cycles (e.g. `A` references `B` and `B` references `A` but both are unreachable), Python runs a cycle-finding algorithm:
1. It copies object reference counts into `gc_refs`.
2. It traverses all tracked objects and decrements the `gc_refs` of any destination object it references.
3. If an object's `gc_refs` drops to 0, it is flagged as a candidate for deletion.
4. Reachable objects are restored along with their dependencies. Unreachable cycles are garbage collected.

---

### 3. NumPy Strides and Memory Address Calculations
A NumPy array is a contiguous block of raw memory. Strides are the number of bytes the CPU must step through memory to advance by 1 element along a given dimension.

For a 2D array of shape \((R, C)\) holding float64 values (8 bytes):
- **C-Contiguous (Row-Major)**: Strides = \((C \times 8, 8)\).
- **Fortran-Contiguous (Column-Major)**: Strides = \((8, R \times 8)\).

#### Element Addressing Formula:
\[\text{Address}(A[i, j]) = \text{BaseAddress} + i \times \text{Stride}[0] + j \times \text{Stride}[1]\]

---

## Part 3: Expert: PyTorch and Custom C++ Extensions

To bypass Python performance bottlenecks, custom mathematical operations are written in C++ and compiled as PyTorch extensions using `pybind11`:

```cpp
#include <torch/extension.h>

// Custom C++ element-wise clipping function
torch::Tensor clamp_tensor(torch::Tensor input, float min_val, float max_val) {
  auto input_contig = input.contiguous();
  auto output = torch::zeros_like(input_contig);
  
  float* input_ptr = input_contig.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();
  int num_elements = input_contig.numel();

  #pragma omp parallel for
  for (int i = 0; i < num_elements; ++i) {
      float val = input_ptr[i];
      output_ptr[i] = (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
  }
  return output;
}

// Bind native code to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("clamp_tensor", &clamp_tensor, "Custom tensor clamping extension");
}
```

---

## Part 4: OP Level: GPU Architecture and CUDA Kernels

GPUs execute calculations across thousands of execution cores in parallel.

### 1. GPU Execution Hierarchy: Grids, Blocks, and Threads
- **Thread**: The base unit executing a single kernel instruction.
- **Thread Block**: A collection of threads executing on a single Streaming Multiprocessor (SM) sharing a common **Shared Memory** space.
- **Grid**: A collection of Blocks spawned for a kernel.

#### Thread Indexing Equation
For a 1D grid layout, each thread calculates its global index:
\[\text{idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}\]

---

### 2. CUDA Memory Optimization
- **Global Memory**: Huge off-chip device memory. High latency. Access must be **coalesced** (threads in a warp reading adjacent addresses in a single transaction).
- **Shared Memory**: On-chip block-scoped cache. Fast, but subject to **bank conflicts** (multiple threads requesting values in the same memory bank, serializing accesses).

---

### 3. Vector Addition CUDA Kernel
This kernel executes in parallel on the GPU, calculating element-wise vector additions:

```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // Compute global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check to avoid illegal memory writes
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```
Host invocation configuration (CPU side code):
```cpp
void run_vector_add(float* h_A, float* h_B, float* h_C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);
    
    // Allocate GPU memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy data to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Grid Launch Setup (256 threads per block)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch Kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Copy results back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Free GPU allocations
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```
