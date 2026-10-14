# NEURAL-NETWORK
A study of neural network and its practise

The system features a custom computer vision pipeline designed to ingest, process, and accurately classify raw, real-world handwritten inputs.

• Input Topology: 64 Features (Flat 8 × 8 Grayscale Image Vectors)

• Hidden Topology: 128 Neurons with Rectified Linear Unit (ReLU) Activation

• Output Topology: 10 Latent Categories via a Numerically Stable Softmax Layer

• Loss Profile: Multi-Class Cross-Entropy


## Mathematical Framework & Vectorization

Unlike traditional scalar loops, this engine operates completely on vectorized batches to optimize system memory access and cache layout.

### Forward Propagation
For a batch input matrix $X \in \mathbb{R}^{M \times 64}$:

$$Z_1 = XW_1 + b_1$$
$$A_1 = \max(0, Z_1)$$
$$Z_2 = A_1W_2 + b_2$$
$$A_2 = \text{Softmax}(Z_2)$$

Where the Softmax activation is structurally modified for numerical stability against floating-point bit limits by isolating the maximum scaling factor:

$$\text{Softmax}(Z)_{i,j} = \frac{e^{Z_{i,j} - \max(Z_i)}}{\sum_{k} e^{Z_{i,k} - \max(Z_i)}}$$

### Vectorized Backpropagation
The matrix gradients tracking the total loss error across dimensional transformations are explicitly mapped without relying on automated tracking configurations:

1. **Output Layer Error Tracking Matrix**
   $$dZ_2 = \frac{A_2 - Y}{M}$$

2. **Hidden to Output Parameter Gradients**
   $$dW_2 = A_1^T \cdot dZ_2 \quad\text{and}\quad db_2 = \sum_{\text{axis } 0} dZ_2$$

3. **Hidden Layer Activation Masking**
   $$dA_1 = dZ_2 \cdot W_2^T$$
   $$dZ_1 = dA_1 \odot \mathbb{I}(Z_1 > 0)$$

4. **Input to Hidden Parameter Gradients**
   $$dW_1 = X^T \cdot dZ_1 \quad\text{and}\quad db_1 = \sum_{\text{axis } 0} dZ_1$$


## Process/Workflow in CV

[Raw Photo Input] 

       │
       ▼
       
[Grayscale Conversion] -> Drops chroma variance noise

       │
       ▼
       
[Dynamic Bounding Box Tracking] -> Locates exact coordinate boundaries via mask (.getbbox())

       │
       ▼
       
[Polarity Inversion] -> Swaps values to match training domain (bright features on dark bg)

       │
       ▼
       
[Aspect-Ratio Uniform Padding] -> Creates a uniform square matrix without distorting the shape

       │
       ▼
       
[Lanczos Downsampling] -> Samples image down to the explicit 8x8 structural footprint
