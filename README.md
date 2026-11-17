# Applied LLM Systems: Fine-Tuning, Scratch RAG & Serving Optimization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/HuggingFace-PEFT%20%26%20LoRA-yellow.svg)](https://github.com/huggingface/peft)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS%20%26%20NumPy-green.svg)](https://github.com/facebookresearch/faiss)
[![Quantization](https://img.shields.io/badge/Inference-GGUF%20%2F%20AWQ%20%2F%20vLLM-purple.svg)](https://github.com/vllm-project/vllm)

> **Candidate:** Harsh Maurya  
> **Repository:** [https://github.com/Advik-harsha/applied-llm-systems](https://github.com/Advik-harsha/applied-llm-systems)  
> **Purpose:** Technical assessment for tndrr AI/ML Internship & applied systems portfolio demonstrating parameter-efficient fine-tuning, frameworkless vector retrieval, and quantized serving optimization.

---

## 📑 Repository Structure & Module Overview

```
applied-llm-systems/
├── task1_finetuning/               # Module 1: LoRA / QLoRA Fine-Tuning (≤ 1.5B Params)
│   ├── dataset/                    # Synthetic & real customer intent extraction (train/test split)
│   ├── train_lora.py               # Hugging Face PEFT/TRL training pipeline
│   ├── evaluate.py                 # Automated evaluation harness (Base vs. Fine-Tuned)
│   ├── failure_analysis.md         # In-depth analysis of 3 edge-case failures & mitigations
│   └── 7b_memory_calculation.md    # Rigorous GPU VRAM derivation (Full FT vs LoRA vs QLoRA)
│
├── task2_rag/                      # Module 2: Retrieval (RAG) Architecture from Scratch
│   ├── documents/                  # 48-page public document (NIST AI Risk Management Framework 1.0)
│   ├── rag_engine.py               # Frameworkless RAG: PDF parsing, chunking, embeddings & FAISS/NumPy
│   ├── benchmark_chunks.py         # Empirical chunk comparison (Recall@5 & MRR across 256 vs 1024 chars)
│   ├── test_dataset.json           # 15 domain-expert test questions with exact ground-truth passages
│   ├── failure_cases_analysis.md   # Analysis of 2 empirical retrieval failures and structural fixes
│   └── bm25_and_reranker_deepdive.md # Theoretical defense: Lexical BM25 vs Dense + Cross-Encoder Re-rankers
│
├── task3_local_optimization/       # Module 3: Local Quantized Inference & Serving Optimization
│   ├── benchmark_runner.py         # Profiler measuring tokens/sec, TTFT, and Peak RSS memory
│   ├── test_prompts.json           # 10 identical diverse prompt benchmarks
│   ├── kv_cache_and_quantization.md# First-principles explanation of KV Cache and INT4 quantization
│   └── 20_user_concurrency_plan_24gb.md # Production serving plan: 20 concurrent users on 1x 24GB GPU
│
├── report/
│   ├── generate_pdf_report.py      # Automated ReportLab PDF generator
│   └── tndrr_AI_Assignment_Report.pdf # Final 4-page executive PDF submission
└── README.md
```

---

## 🚀 Module 1 — Fine-Tuning with LoRA/QLoRA

### 1. Task Definition & Model Selection
* **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (490M parameters). Chosen for fast convergence on consumer/Colab T4 GPUs while exhibiting strong instruct priors.
* **Narrow Domain Task:** **Natural Language Customer Support Intent to Strict Actionable JSON Schema**.
  * *Input:* Unstructured customer message requesting refunds, exchanges, flight modifications, or delivery complaints.
  * *Target:* Deterministic JSON conforming to `{intent, order_id, item_name, issue_type, details, requested_resolution, refund_amount, currency}`.
* **Why this benchmark:** Un-tuned base models suffer from conversational bloat, markdown formatting noise, hallucinated extra keys, or schema invalidity. Fine-tuning aligns the model to act as a deterministic extraction API.

### 2. Empirical Results on Held-Out Test Set ($N = 60$)
| Model Variant | JSON Syntactic Validity (%) | Schema Exact Key Match (%) | Latency / Sample (ms) |
| :--- | :---: | :---: | :---: |
| **Qwen2.5-0.5B Base (Zero-Shot)** | 41.7% | 48.3% | 340 ms |
| **Qwen2.5-0.5B + LoRA (Fine-Tuned)** | **98.3%** | **96.7%** | **315 ms** |

### 3. LoRA Hyperparameter Configuration
* **Rank ($r = 16$) & Alpha ($\alpha = 32$):** Scaling factor $\alpha / r = 2.0$. Rank 16 provides sufficient parameter capacity for structural grammar alignment without overfitting on limited samples.
* **Target Modules:** All linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Adapting both attention and MLP feed-forward networks guarantees superior representation transfer compared to attention-only adaptation.
* **Learning Rate:** $2 \times 10^{-4}$ with Cosine Annealing scheduler and 5% warmup.
* **Training Budget:** 3 epochs, batch size 4, gradient accumulation steps 2 on single T4 GPU (~7 minutes).

### 4. GPU Memory Breakdown for 7B Model Fine-Tuning
A complete mathematical derivation of peak VRAM requirements during 16-bit and 4-bit fine-tuning of a 7-Billion parameter transformer ($N = 7 \times 10^9$):

$$\text{Memory}_{\text{total}} = M_{\text{weights}} + M_{\text{grads}} + M_{\text{optimizer}} + M_{\text{activations}} + M_{\text{overhead}}$$

| Component | (a) Full Fine-Tuning (16-bit Mixed) | (b) LoRA (16-bit Base + Adapters) | (c) QLoRA (4-bit NF4 + Adapters) |
| :--- | :--- | :--- | :--- |
| **Base Model Weights** | $7\text{B} \times 2\text{B} = \mathbf{14.0\text{ GB}}$ | $7\text{B} \times 2\text{B} = \mathbf{14.0\text{ GB}}$ | $7\text{B} \times 0.5\text{B} + \text{quant overhead} \approx \mathbf{3.8\text{ GB}}$ |
| **Adapter Weights** | N/A | $\approx 20\text{M} \times 2\text{B} \approx \mathbf{0.04\text{ GB}}$ | $\approx 20\text{M} \times 2\text{B} \approx \mathbf{0.04\text{ GB}}$ |
| **Gradients** | $7\text{B} \times 2\text{B} = \mathbf{14.0\text{ GB}}$ | Only adapters: $\approx \mathbf{0.04\text{ GB}}$ | Only adapters: $\approx \mathbf{0.04\text{ GB}}$ |
| **Optimizer States (AdamW)** | $7\text{B} \times 12\text{B} = \mathbf{84.0\text{ GB}}$ (FP32 master, momentum, variance) | Only adapters: $20\text{M} \times 12\text{B} \approx \mathbf{0.24\text{ GB}}$ | Paged AdamW 8-bit on adapters: $\approx \mathbf{0.16\text{ GB}}$ (pages to CPU if needed) |
| **Activations (w/ Grad Checkpointing)** | $\approx \mathbf{4.5\text{ GB}}$ ($b=2, s=2048$) | $\approx \mathbf{2.5\text{ GB}}$ ($b=2, s=2048$) | $\approx \mathbf{1.8\text{ GB}}$ ($b=2, s=2048$) |
| **CUDA Buffers & Context** | $\approx \mathbf{2.0\text{ GB}}$ | $\approx \mathbf{1.5\text{ GB}}$ | $\approx \mathbf{1.2\text{ GB}}$ |
| **Total Peak VRAM Required** | **$\approx 118.5\text{ GB}$** *(Needs $\ge 2\times$ 80GB A100)* | **$\approx 18.3\text{ GB}$** *(Fits on 1x 24GB RTX 3090/4090)* | **$\approx 7.0\text{ GB}$** *(Fits on free 15GB Colab T4 or 8GB RTX 4060)* |

---

## 🔍 Module 2 — Retrieval (RAG) Architecture from Scratch

### 1. Frameworkless Design
* Built strictly without high-level wrappers like LangChain or LlamaIndex.
* Core pipeline implemented in pure modular Python:
  1. **Document Ingestion:** PDF text extraction with PyMuPDF (`fitz`) / `pypdf`.
  2. **Sliding Window Chunking:** Configurable character/token chunking with overlap.
  3. **Dense Vector Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`, 384-dimensional).
  4. **Vector Similarity Index:** Pure NumPy inner-product cosine similarity & FAISS `IndexFlatIP`.
  5. **Context Injection & Generation:** Prompt formatting and response generation.

### 2. Empirical Benchmark: Chunk Size Comparison
Evaluated on **NIST AI Risk Management Framework 1.0 (48 pages)** using 15 hand-crafted domain questions:

| Chunking Configuration | Chunk Size (Chars) | Overlap (Chars) | Total Chunks | Recall@5 | MRR@5 | Mean Query Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Setting A (Small Chunks)** | 256 | 50 | 412 | 66.7% (10/15) | 0.528 | 4.8 ms |
| **Setting B (Large Chunks)** | **1024** | **150** | **118** | **93.3% (14/15)** | **0.844** | **3.9 ms** |

* **Why Setting B Outperformed Setting A:**
  1. *Contextual Coherence:* Policy and risk definitions in technical governance frameworks span 3–6 sentences. Chunks of 256 characters frequently broke definitions mid-sentence, causing key noun phrases to be separated from their qualifiers.
  2. *Higher Semantic Density:* Dense sentence-transformer embeddings encode broader topical paragraphs with stronger semantic signals than fragment vectors, resulting in higher top-1 rank precision.

---

## ⚡ Module 3 — Local Model Running & Optimization

### 1. Latency & Memory Profiling (FP16 vs. 4-bit Quantization)
Benchmarked on identical 10 multi-domain prompts across instruction following, reasoning, code generation, and factual retrieval:

| Metric | Setting 1: Unquantized Baseline (FP16 / Q8_0) | Setting 2: Quantized Model (Q4_K_M / INT4) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Model Footprint / RAM** | 1,020 MB | 362 MB | **2.82× Memory Savings** |
| **Prompt Eval Speed** | 38.4 tokens/sec | 64.2 tokens/sec | **1.67× Faster** |
| **Generation Speed** | 22.1 tokens/sec | 41.8 tokens/sec | **1.89× Throughput Boost** |
| **Time to First Token (TTFT)** | 145 ms | 88 ms | **39.3% Latency Reduction** |
| **Qualitative Output Quality** | High syntactic accuracy; full verbosity | 98% parity with baseline; minor style differences on creative tasks | Negligible degradation on logic & code |

### 2. Production Concurrency Plan: 20 Users on 1x 24GB GPU
* **Architecture:** `vLLM` serving **Llama-3.1-8B-Instruct (AWQ 4-bit)** with continuous iteration-level batching and PagedAttention.
* **Weights Memory:** $8\text{B} \times 0.55\text{ bytes} \approx \mathbf{4.5\text{ GB}}$.
* **Runtime Activations & Context:** $\approx \mathbf{2.0\text{ GB}}$.
* **Available VRAM for KV Cache:** $24\text{ GB} - 4.5 - 2.0 = \mathbf{17.5\text{ GB}}$.
* **KV Cache Per User (GQA: 32 layers, 8 KV heads, head dim 128, FP16):**
  $$\text{KV per token} = 2 \times 32 \times 8 \times 128 \times 2\text{ bytes} = 131,072\text{ bytes} \approx 128\text{ KB/token}$$
  $$\text{At } 2048\text{ context tokens: } 2048 \times 128\text{ KB} = \mathbf{256\text{ MB per concurrent user}}$$
* **Total KV Cache for 20 Concurrent Users:** $20 \times 256\text{ MB} = \mathbf{5.12\text{ GB}}$.
* **Total Peak VRAM Used:**
  $$\text{Total} = 4.5\text{ GB (Weights)} + 5.12\text{ GB (KV)} + 2.0\text{ GB (Activations)} = \mathbf{11.62\text{ GB}} \ll 24\text{ GB}$$
  * *Headroom:* $\mathbf{12.38\text{ GB}}$ remaining, comfortably supporting burst spikes to 40+ concurrent sessions or $4096+$ token context windows.

---

## 📄 Submission Report
The final executive report is generated via ReportLab and compiled to [report/tndrr_AI_Assignment_Report.pdf](report/tndrr_AI_Assignment_Report.pdf).
It adheres strictly to the 4-page ceiling and contains all empirical tables, formulas, and code snippets.
