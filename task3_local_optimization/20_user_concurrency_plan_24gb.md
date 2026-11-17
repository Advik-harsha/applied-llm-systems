# Task 3: Production Serving Plan — 20 Concurrent Users on One 24 GB GPU

---

## 1. Executive Summary & Objective

**Goal:** Design an end-to-end inference architecture capable of concurrently serving **20 active users** with interactive conversational response times on a single **24 GB GPU** (e.g., NVIDIA GeForce RTX 3090 / RTX 4090, or enterprise NVIDIA A10G / L4).

---

## 2. Explicit System & Operational Assumptions

1. **Hardware Specification:**
   * GPU: Single NVIDIA RTX 4090 (24 GB GDDR6X, 1,008 GB/s memory bandwidth) or NVIDIA A10G (24 GB GDDR6, 600 GB/s bandwidth).
   * Total Addressable VRAM: $24.0\text{ GB} = 24,576\text{ MB}$.
2. **Model Selection:**
   * **Meta Llama-3.1-8B-Instruct** (or **Qwen2.5-7B-Instruct**).
   * Quantization Format: **4-bit AWQ (Activation-aware Weight Quantization)** or **GPTQ INT4**.
   * Model Parameters: $N = 8.03 \times 10^9$.
   * Architecture: Transformer decoder with **Grouped-Query Attention (GQA)**:
     * Layers ($n_{\text{layers}}$): $32$
     * Hidden Dimension ($d_{\text{model}}$): $4096$
     * Query Heads ($n_{\text{heads}}$): $32$
     * Key/Value Heads ($n_{\text{kv\_heads}}$): **$8$** (GQA reduces KV cache size by $4\times$ compared to MHA!)
     * Head Dimension ($d_{\text{head}}$): $128$
3. **Workload & Traffic Profile:**
   * Active Concurrency: $B = 20\text{ concurrent users}$.
   * Average Context Length per User ($L$): **$2,048\text{ tokens}$** ($1,500\text{ input prompt tokens} + 548\text{ output generation tokens}$).
   * Acceptable Generation Latency: $\ge 25\text{ tokens/sec per user}$ (faster than average human reading speed).
4. **Serving Engine:**
   * **vLLM** running **PagedAttention** with dynamic **Continuous Batching** (iteration-level scheduling).
   * OS/CUDA Buffer Reservation: $1.5 - 2.0\text{ GB}$.

---

## 3. Mathematical VRAM Budget Breakdown

Total available GPU memory is budgeted into three primary categories:

$$VRAM_{\text{total}} = VRAM_{\text{model}} + VRAM_{\text{runtime/activations}} + VRAM_{\text{kv\_cache}}$$

---

### Step 1: Base Model Weights Memory ($VRAM_{\text{model}}$)
* In 4-bit AWQ, each weight occupies 4 bits ($0.5\text{ bytes}$).
* Incorporating quantization metadata (scaling factors and zero-points per group size 128 $\approx 0.05\text{ bytes/param}$):
  $$VRAM_{\text{model}} = 8.03 \times 10^9 \times 0.55\text{ bytes} \approx \mathbf{4.42\text{ GB}} \quad (\text{Reserve } \mathbf{4.80\text{ GB}})$$

---

### Step 2: Runtime Activations & CUDA Overhead ($VRAM_{\text{runtime}}$)
* CUDA kernel context, cuBLAS workspaces, and temporary execution scratchpads: $\approx 1.50\text{ GB}$.
* Transient intermediate activation tensors during the prefill phase: $\approx 0.70\text{ GB}$.
* Total Runtime Reservation:
  $$VRAM_{\text{runtime}} = \mathbf{2.20\text{ GB}}$$

---

### Step 3: Available VRAM for Dynamic KV Cache Pool
Subtracting model weights and runtime overhead from total capacity:

$$VRAM_{\text{available for KV}} = 24.00\text{ GB} - 4.80\text{ GB} - 2.20\text{ GB} = \mathbf{17.00\text{ GB}} \quad (17,408\text{ MB})$$

---

### Step 4: KV Cache Memory Formula & Per-User Allocation
For a model utilizing Grouped-Query Attention (GQA), the memory required to store the Key and Value vectors for **one single token** across all layers is:

$$\text{Memory}_{\text{per\_token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_element}$$

Where:
* Factor of $2$: One tensor for Key, one tensor for Value.
* $n_{\text{layers}} = 32$.
* $n_{\text{kv\_heads}} = 8$ (GQA).
* $d_{\text{head}} = 128$.
* $\text{bytes\_per\_element} = 2\text{ bytes}$ (16-bit FP16 precision).

$$\text{Memory}_{\text{per\_token}} = 2 \times 32 \times 8 \times 128 \times 2\text{ bytes} = 131,072\text{ bytes} = \mathbf{128\text{ KB per token}}$$

*(Notice: If Multi-Head Attention without GQA were used with 32 KV heads, it would require $512\text{ KB/token}$, consuming $4\times$ more memory!)*

#### KV Cache Required Per User at $L = 2,048$ tokens:
$$\text{KV}_{\text{user}} = 2,048\text{ tokens} \times 128\text{ KB/token} = 262,144\text{ KB} = \mathbf{256\text{ MB per user}}$$

#### Total KV Cache Required for 20 Concurrent Users:
$$VRAM_{\text{KV\_20\_users}} = 20 \times 256\text{ MB} = 5,120\text{ MB} = \mathbf{5.00\text{ GB}}$$

---

## 4. Final Memory Allocation & Safety Headroom

| Subsystem Component | Allocated VRAM (GB) | Share of Total (%) | Notes |
| :--- | :---: | :---: | :--- |
| **Model Weights (Llama-3.1-8B AWQ 4-bit)** | **4.80 GB** | 20.0% | Static 4-bit weight footprint |
| **CUDA Runtime & Execution Activations** | **2.20 GB** | 9.2% | Kernels, cuDNN, prefill buffers |
| **KV Cache Pool (20 Concurrent Users @ 2K Context)** | **5.00 GB** | 20.8% | $20 \times 256\text{ MB}$ GQA FP16 |
| **Total Peak Utilized VRAM** | **12.00 GB** | **50.0%** | **Fits with 50% headroom!** |
| **Free / Elastic Buffer Headroom** | **12.00 GB** | **50.0%** | Dedicated buffer for traffic spikes |

---

## 5. Architectural Implementation Details

### 1. PagedAttention (Eliminating Memory Fragmentation)
In conventional inference implementations (Hugging Face Transformers), memory for the maximum sequence length must be pre-allocated contiguously. Because actual request lengths vary, this wastes **60% to 80%** of GPU memory due to internal and external fragmentation.
* **Our Plan:** Deploy **vLLM with PagedAttention**. 
* PagedAttention partitions the KV cache into fixed-size physical blocks (e.g., 16 tokens per block) allocated on demand in non-contiguous memory, exactly like Virtual Memory paging in modern OS kernels.
* **Result:** Near-zero memory waste ($< 4\%$), allowing the 5.0 GB KV budget to be utilized with 96%+ efficiency.

### 2. Continuous (Iteration-Level) Batching
Rather than waiting for an entire batch of 20 requests to complete generation before scheduling new ones:
* **Iteration-Level Scheduling:** When User #3 finishes generating their 150-token response early, their KV cache blocks are instantly reclaimed, and User #21's waiting prompt is dynamically inserted into the very next forward iteration.
* **Latency Guarantee:** Eliminates head-of-line blocking, maintaining steady $> 35\text{ tok/sec}$ throughput across all 20 active connections.

### 3. Traffic Spike & Context Overflow Strategy
What happens if users burst to $4,096\text{ context tokens}$ or concurrency spikes to 40 users?
* Because our baseline plan consumes only **$12.00\text{ GB}$ of the $24.0\text{ GB}$ capacity**, the system has **$12.00\text{ GB}$ of unallocated elastic headroom**.
* **Burstable Capacity:**
  * Maximum concurrent users at $2,048$ tokens: $\frac{17.0\text{ GB KV Pool}}{0.256\text{ GB/user}} \approx \mathbf{66\text{ concurrent users}}$!
  * If context extends to $8,192\text{ tokens}$ ($1.0\text{ GB/user}$), the 24 GB card still comfortably sustains $\approx 16\text{ concurrent users}$ without out-of-memory errors!
* **KV Cache Quantization (Optional Fallback):** If traffic exceeds 60 users, enable **FP8 KV Caching** (`--kv-cache-dtype fp8`), cutting KV memory per token in half from $128\text{ KB}$ to $64\text{ KB}$, doubling concurrency to **130+ users**.
