# Task 2: Empirical Retrieval Failure Analysis & Structural Fixes

Our benchmark evaluated retrieval on the official 86-page **NIST AI Risk Management Framework 1.0 (NIST.SP.1270)** across two chunking strategies (Setting A: 256 characters vs. Setting B: 1024 characters). 

Below is an in-depth diagnosis of two real retrieval failures observed in our empirical runs, along with concrete architectural remedies.

---

### Failure Case 1: Caption & Taxonomy Fragmentation (Observed in Setting A - Small Chunks)

* **Query (Question `q05`):**
  > *"What are the three major overarching categories of AI bias outlined in the NIST taxonomy?"*
* **Ground Truth Target Passage (Page 17, Section 1.2):**
  > *"Categories of AI Bias: Systemic, Statistical and computational, and Human cognitive biases. Systemic bias results from historical and institutional practices; statistical bias stems from unrepresentative data; and human bias involves cognitive heuristics."*
* **Retrieved Output in Setting A:**
  * Top-1 Chunk (Page 9): Introductory discussion on bias definitions.
  * Top-2 Chunk (Page 15): Fragment discussing human nonverbal traits.
  * Top-3 Chunk (Page 18): Fragment discussing statistical error.
  * Ground truth rank: **Missed Top-5 (Rank > 5)**.
* **Root Cause Diagnosis (Semantic Fragmentation):**
  * In Setting A (256 characters, 50 overlap), the taxonomy explanation on Page 17 was split into **3 separate adjacent chunks**:
    * Fragment 1: *"Fig. 2. Categories of AI Bias. The leaf node terms in each subcategory in the picture are hyperlinked..."*
    * Fragment 2: *"to the GLOSSARY. Clicking them will bring up the definition in the Glossary. Systemic bias results from..."*
    * Fragment 3: *"historical and institutional practices; statistical bias stems from unrepresentative data; and human bias..."*
  * Because the three categories (`Systemic`, `Statistical`, and `Human`) were split across separate fragments, **no single chunk vector contained the complete triad**. 
  * As a result, the embedding similarity score was severely diluted, and none of the fragments broke into the top-5 results.
* **Architectural Remedy:**
  1. **Hierarchical / Parent-Child Chunking:** Index smaller child chunks (256 chars) for dense vector search, but return the enclosing **parent chunk** (1024–2048 chars) or full section to the LLM context window.
  2. **Structure-Aware Chunking (Markdown / Heading-based splitting):** Instead of naive fixed-character windows, split by structural headings (`## Section 1.2`) or table/figure blocks, ensuring captions and taxonomy summaries remain intact.

---

### Failure Case 2: Information Dilution / Needle-in-a-Haystack (Observed in Setting B - Large Chunks)

* **Query (Question `q10`):**
  > *"How can the choice of an objective function introduce bias into a machine learning model?"*
* **Ground Truth Target Passage (Page 32, Section 2.1):**
  > *"Furthermore, the choice of the model's objective function, upon which a model's definition of accuracy is based, can reflect bias. Not taking context into consideration during model selection can lead to biased results for sub-populations."*
* **Retrieved Output in Setting B:**
  * Top-1 to Top-5 Chunks (Pages 29, 30, 36, 37): Large chunks discussing general TEVV protocols, validation datasets, and metric definitions.
  * Ground truth chunk (Page 32): **Rank 7 (Missed Top-5)**.
* **Root Cause Diagnosis (Topic Dilution):**
  * In Setting B (1024 characters), the two sentences concerning *"the choice of the model's objective function"* were embedded alongside 800 characters of broad, general text discussing model accuracy, benchmarking datasets, and operational environments.
  * When dense embeddings are generated over a long text block, the resulting vector represents the **centroid (average semantic topic)** of the entire 1024 characters.
  * Because the query specifically targeted *"objective function bias"*, the broad centroid vector for the Page 32 chunk had lower cosine similarity than other chunks where related keywords happened to appear with higher relative density.
* **Architectural Remedy:**
  1. **Hybrid Dense + Sparse Search (BM25 + Dense Vectors):** The phrase *"objective function"* is an exact, highly informative technical term. While dense embeddings diluted it, BM25 assigns high inverse document frequency (IDF) weight to *"objective function"*, immediately pulling this passage into the top 3.
  2. **Cross-Encoder Re-Ranking:** Retrieve the top-20 candidates using vector search, then pass each candidate with the query into a cross-encoder model (e.g., `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`). The cross-encoder's full token-to-token attention detects the exact clause and elevates it to Rank 1.
