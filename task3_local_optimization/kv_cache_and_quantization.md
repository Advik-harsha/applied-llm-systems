# Task 3: The KV Cache & Why Quantization Saves Memory

*(A rigorous first-principles engineering derivation of inference memory dynamics, KV cache mechanics, and weight-activation quantization)*

---

## 1. What the KV Cache Is & Why It Is Essential

### The Autoregressive Decoding Problem
In modern Transformer-based autoregressive Large Language Models (LLMs), generation proceeds **token by token**. To predict the probability distribution for token $x_{t+1}$, the self-attention mechanism must compute relationships between the new token and all preceding tokens $x_1, x_2, \dots, x_t$.

The scaled dot-product attention formula at each attention head is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Where:
* $Q \in \mathbb{R}^{S \times d_k}$: Queries projected from the input.
* $K \in \mathbb{R}^{S \times d_k}$: Keys representing the sequence tokens.
* $V \in \mathbb{R}^{S \times d_k}$: Values representing the token embeddings.

### Without the KV Cache ($O(t^2)$ Waste):
If we do not cache past representations, at step $t=1$, we compute $Q_1, K_1, V_1$.
At step $t=2$, we feed tokens $[x_1, x_2]$ into the network, recalculating $K_1, V_1$ from scratch alongside $K_2, V_2$.
At step $t=1000$, we would re-run the entire transformer across all 1,000 previous tokens just to predict token 1,001!
* **Computational Cost:** Cumulative compute scales quadratically as $\sum_{i=1}^{T} i = \frac{T(T+1)}{2} = \mathcal{O}(T^2)$.
* For interactive generation, this quickly causes catastrophic latency slowdowns.

### With the KV Cache ($O(1)$ per step):
Because previously generated tokens never change, their projected Key ($K$) and Value ($V$) tensors at every layer remain **static**.
* Instead of recomputing them, we store the historical $K$ and $V$ tensors across all $L$ layers in GPU VRAM:
  $$\mathbf{K}_{\text{cached}} = [K_1, K_2, \dots, K_t], \quad \mathbf{V}_{\text{cached}} = [V_1, V_2, \dots, V_t]$$
* At step $t+1$, we only pass the **single newly generated token** $x_{t+1}$ into the model.
* The model computes only a single query row $Q_{t+1}$, a single key row $K_{t+1}$, and a single value row $V_{t+1}$.
* We concatenate $K_{t+1}$ and $V_{t+1}$ to the cache, compute attention:
  $$\text{Attention}_{t+1} = \text{softmax}\left(\frac{Q_{t+1} \mathbf{K}_{\text{cached}}^\top}{\sqrt{d_k}}\right) \mathbf{V}_{\text{cached}}$$
* **Computational Cost:** Per-token decoding drops from $\mathcal{O}(t)$ to $\mathcal{O}(1)$ constant time!

### The Engineering Trade-off: Memory vs. Compute
While the KV cache saves massive compute, it transfers the operational bottleneck from **compute bound** to **memory capacity and memory bandwidth bound**. For long context lengths ($> 8\text{K}$) and high batch sizes, the KV cache frequently exceeds the size of the model weights themselves!

---

## 2. Why Quantization Saves Memory & Speeds Up Generation

Quantization is the process of mapping high-precision continuous floating-point weights to lower-bit discrete integer representations:

$$\mathbf{w}_{\text{quant}} = \text{round}\left( \frac{\mathbf{w}}{\text{scale}} \right) + \text{zero\_point}$$

### 1. Direct Memory Capacity Reduction
Standard neural networks store weights in 16-bit floating point (`FP16` or `BF16`), which requires **2 bytes (16 bits) per parameter**.
* In **INT4** or **Q4_K_M** quantization, each weight is represented using only **4 bits (0.5 bytes)**:
  $$\text{Memory Compression Factor} = \frac{16\text{ bits}}{4\text{ bits}} = \mathbf{4\times \text{ Reduction}}$$
* A 7-Billion parameter model:
  * In FP16: $7\text{B} \times 2\text{ bytes} = \mathbf{14.0\text{ GB}}$
  * In 4-bit (INT4 / AWQ): $7\text{B} \times 0.5\text{ bytes} \approx \mathbf{3.5 - 4.0\text{ GB}}$
* This allows massive models that previously required enterprise $80\text{GB}$ GPUs to fit entirely within consumer VRAM ($8\text{GB} - 24\text{GB}$) or edge device RAM.

---

### 2. Why Quantization Dramatically Increases Tokens Per Second
A common misconception is that quantization makes computation faster by doing simpler integer arithmetic. In reality, modern inference engines often dequantize 4-bit weights back to 16-bit on-the-fly during matrix multiplication!

**Why does generation speed up then?**
Because LLM autoregressive token generation is strictly **Memory Bandwidth Bound**, not Compute Bound.

* **The Arithmetic Intensity Reality:**
  During single-token decoding (batch size 1), the model must stream **every single parameter** from GPU high-bandwidth memory (HBM) into the GPU streaming multiprocessors (SMs) across the PCIe/memory bus just to produce **one token**:
  $$\text{Operational Intensity} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Bytes Moved from Memory}}$$
* In generation mode, the operational intensity is extremely low ($\approx 1\text{ FLOP/byte}$). The GPU compute tensor cores spend most of their time idle, waiting for bytes to arrive over the memory bus.
* **The Quantization Advantage:**
  When weights are compressed $4\times$ (from 16-bit to 4-bit), **the memory bus only needs to transfer one-fourth the number of bytes per token**:
  $$\text{Throughput} \propto \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Size (GB)}}$$
  Moving 4 GB of data across the memory bus takes one-fourth the time of moving 16 GB, resulting in near-linear increases in tokens-per-second generation speed on consumer GPUs and CPUs!
