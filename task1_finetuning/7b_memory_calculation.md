# Task 1: Theoretical GPU Memory Calculation for a 7B Parameter Model

This document presents a rigorous first-principles calculation of the peak GPU memory (VRAM) required to fine-tune a **7-Billion parameter ($N = 7 \times 10^9$) Transformer model** under three paradigms:
1. **(a) Full Fine-Tuning** (16-bit mixed precision with AdamW)
2. **(b) LoRA** (Low-Rank Adaptation, 16-bit frozen base model + 16-bit trainable adapters)
3. **(c) QLoRA** (Quantized LoRA, 4-bit NF4 frozen base model + 16-bit adapters + Paged Optimizer)

---

## 1. Mathematical Breakdown of Memory Allocation

The peak memory consumed during model training is governed by:

$$M_{\text{peak}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}} + M_{\text{overhead}}$$

Where:
* $M_{\text{weights}}$: Memory required to store the static model parameters.
* $M_{\text{gradients}}$: Memory to store gradients calculated during the backward pass.
* $M_{\text{optimizer}}$: Memory required by the optimizer to track historical moments.
* $M_{\text{activations}}$: Transient tensors saved during the forward pass for backpropagation.
* $M_{\text{overhead}}$: CUDA runtime context, PyTorch memory fragmentation allocator reserve, and temporary communication buffers.

---

## 2. Derivation by Paradigm

### (a) Full Fine-Tuning (16-bit Mixed Precision with AdamW)
All $N = 7 \times 10^9$ parameters are updated.

1. **Model Weights ($M_{\text{weights}}$):**
   * Stored in 16-bit (FP16 or BF16) = $2\text{ bytes per parameter}$.
   * $M_{\text{weights}} = 7 \times 10^9 \times 2\text{ bytes} = 14.0\text{ GB}$.

2. **Gradients ($M_{\text{gradients}}$):**
   * Gradients are computed and stored in 16-bit for every parameter.
   * $M_{\text{gradients}} = 7 \times 10^9 \times 2\text{ bytes} = 14.0\text{ GB}$.

3. **Optimizer States ($M_{\text{optimizer}}$ - Standard AdamW):**
   * Standard AdamW maintains 32-bit (FP32) master weights and two 32-bit momentum tracking buffers for numerical stability:
     * FP32 Master Weight Copy: $4\text{ bytes/param}$
     * FP32 First Moment (running mean of gradients, $m_t$): $4\text{ bytes/param}$
     * FP32 Second Moment (running variance of gradients, $v_t$): $4\text{ bytes/param}$
     * Total = $12\text{ bytes per parameter}$.
   * $M_{\text{optimizer}} = 7 \times 10^9 \times 12\text{ bytes} = 84.0\text{ GB}$.
   *(Note: Even with 8-bit AdamW, optimizer states require $4\text{ bytes/param} = 28.0\text{ GB}$)*.

4. **Activations ($M_{\text{activations}}$):**
   * **Without Gradient Checkpointing:** Storing intermediate activation tensors for all 32 layers requires:
     $$M_{\text{act\_no\_ckpt}} \approx b \cdot s \cdot h \cdot L \cdot \left(34 + 5 \cdot \frac{a \cdot s}{h}\right) \text{ bytes}$$
     For $b=2, s=2048, h=4096, L=32, a=32$: $\mathbf{M_{\text{act\_no\_ckpt}} \approx 38.6\text{ GB}}$!
   * **With Activation (Gradient) Checkpointing:** Only the input activation to each transformer block is retained during the forward pass; all intermediate layer activations are recomputed on-the-fly during backpropagation:
     $$M_{\text{act\_ckpt}} \approx 2 \cdot b \cdot s \cdot h \cdot L \cdot \text{bytes} + M_{\text{attn\_scratchpad}} \approx 2.15\text{ GB} + 2.35\text{ GB} = \mathbf{4.50\text{ GB}}$$
     *(Enabling gradient checkpointing trades ~20% compute time for an 88% reduction in activation memory!)*

5. **CUDA Runtime Overhead ($M_{\text{overhead}}$):**
   * CUDA kernels, cuDNN handles, and allocator reserve $\approx 2.0\text{ GB}$.

$$\mathbf{M_{\text{Full FT}}} = 14.0\text{ GB} + 14.0\text{ GB} + 84.0\text{ GB} + 4.5\text{ GB} + 2.0\text{ GB} \approx \mathbf{118.5\text{ GB}}$$

> **Hardware Conclusion:** Cannot run on any single consumer or enterprise GPU. Requires at least **$2\times 80\text{GB}$ NVIDIA A100/H100 GPUs** or **$4\times 40\text{GB}$ GPUs** running Fully Sharded Data Parallel (FSDP) or DeepSpeed ZeRO-3.

---

### (b) LoRA (16-bit Frozen Base + 16-bit Trainable Adapters)
Base model weights are frozen. Only low-rank decomposition matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ are trained.
For standard rank $r=16$ applied across all linear layers, trainable adapter parameters $N_{\text{adapter}} \approx 20\text{ Million}$ ($\approx 0.28\%$ of total model size).

1. **Base Model Weights ($M_{\text{weights}}$):**
   * Frozen in 16-bit: $7\text{B} \times 2\text{ bytes} = 14.0\text{ GB}$.

2. **Adapter Weights ($M_{\text{adapter}}$):**
   * $20 \times 10^6 \times 2\text{ bytes} = 0.04\text{ GB}$ ($40\text{ MB}$).

3. **Gradients ($M_{\text{gradients}}$):**
   * Calculated **only** for trainable adapter parameters:
   * $M_{\text{gradients}} = 20 \times 10^6 \times 2\text{ bytes} = 0.04\text{ GB}$ ($40\text{ MB}$).

4. **Optimizer States ($M_{\text{optimizer}}$ - AdamW):**
   * Stored only for adapter parameters ($12\text{ bytes/param}$):
   * $M_{\text{optimizer}} = 20 \times 10^6 \times 12\text{ bytes} = 0.24\text{ GB}$ ($240\text{ MB}$).

5. **Activations ($M_{\text{activations}}$):**
   * With gradient checkpointing ($b=2, s=2048$): $\approx 2.5\text{ GB}$.

6. **CUDA Runtime Overhead:**
   * $\approx 1.5\text{ GB}$.

$$\mathbf{M_{\text{LoRA}}} = 14.0\text{ GB} + 0.04\text{ GB} + 0.04\text{ GB} + 0.24\text{ GB} + 2.5\text{ GB} + 1.5\text{ GB} \approx \mathbf{18.32\text{ GB}}$$

> **Hardware Conclusion:** Fits comfortably on a single **24 GB GPU** (e.g., consumer **NVIDIA RTX 3090 / RTX 4090**, or cloud **NVIDIA A10G / L4**).

---

### (c) QLoRA (4-bit NF4 Quantized Base + 16-bit Adapters + Paged Optimizer)
Base model weights are compressed to 4-bit NormalFloat (NF4) with Double Quantization (DQ), while adapter matrices remain in 16-bit FP16/BF16.

1. **Base Model Weights ($M_{\text{weights}}$):**
   * 4 bits = $0.5\text{ bytes per parameter}$.
   * Theoretical: $7\times 10^9 \times 0.5\text{ bytes} = 3.50\text{ GB}$.
   * Including quantization constants and Double Quantization overhead ($\approx 0.03\text{ bits/param}$): $M_{\text{weights}} \approx 3.80\text{ GB}$.

2. **Adapter Weights + Gradients + Optimizer States:**
   * $M_{\text{adapter}} + M_{\text{grad}} + M_{\text{opt}} \approx 0.04\text{ GB} + 0.04\text{ GB} + 0.16\text{ GB} \approx 0.24\text{ GB}$.
   * Uses *Paged Optimizer* (bitsandbytes `PagedAdamW8bit`), which automatically pages optimizer states to CPU system RAM during memory pressure spikes, preventing out-of-memory errors.

3. **Activations ($M_{\text{activations}}$):**
   * Base weights are dequantized on-the-fly to 16-bit during forward compute; activations with gradient checkpointing ($b=2, s=2048$): $\approx 1.8\text{ GB}$.

4. **CUDA Runtime Overhead:**
   * $\approx 1.1\text{ GB}$.

$$\mathbf{M_{\text{QLoRA}}} = 3.80\text{ GB} + 0.24\text{ GB} + 1.80\text{ GB} + 1.10\text{ GB} \approx \mathbf{6.94\text{ GB}} - \mathbf{7.5\text{ GB}}$$

> **Hardware Conclusion:** Runs effortlessly on a single **free Google Colab T4 GPU (15 GB VRAM)** or an affordable **8 GB consumer GPU** (e.g., NVIDIA RTX 3070 / RTX 4060 with batch size 1).

---

## 3. Summary Comparison Table

| Metric / Parameter | Full Fine-Tuning | LoRA | QLoRA |
| :--- | :---: | :---: | :---: |
| **Base Weight Precision** | 16-bit (FP16/BF16) | 16-bit (FP16/BF16) | 4-bit (NF4 + DQ) |
| **Adapter Precision** | N/A | 16-bit | 16-bit |
| **Trainable Parameters** | 7.0 Billion (100%) | ~20 Million (0.28%) | ~20 Million (0.28%) |
| **Static Model VRAM** | 14.0 GB | 14.0 GB | 3.8 GB |
| **Optimizer Memory** | 84.0 GB | 0.24 GB | 0.16 GB (Paged) |
| **Peak VRAM Needed** | **~118.5 GB** | **~18.3 GB** | **~7.0 GB** |
| **Target Hardware** | $2\times$ 80GB A100 | $1\times$ 24GB RTX 3090/4090 | $1\times$ 15GB Colab T4 / 8GB RTX 4060 |
