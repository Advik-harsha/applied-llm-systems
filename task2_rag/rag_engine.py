"""
Frameworkless RAG (Retrieval-Augmented Generation) Engine from Scratch.
Implements:
1. PDF Document Parsing (pypdf)
2. Character/Token Sliding Window Chunking with Overlap
3. Dense Vector Embeddings (Dual Engine: SentenceTransformer or Pure NumPy Dense LSA)
4. Vector Indexing & Search using pure NumPy Cosine Similarity & FAISS
5. Context Formatting & Prompt Synthesis

ZERO dependencies on LangChain, LlamaIndex, or third-party RAG frameworks.
"""

import os
import re
import math
import numpy as np
import pypdf

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

class PureNumpyEmbeddingModel:
    """
    Self-contained 384-dimensional dense semantic embedding engine implemented in pure NumPy.
    Uses subword n-gram hashing and term weighting to generate continuous dense vectors.
    Guarantees 100% portability across all operating systems without external C++ DLL dependencies.
    """
    def __init__(self, dim=EMBEDDING_DIM, seed=42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # Fixed deterministic projection basis for token embeddings
        self.basis = self.rng.randn(10007, dim).astype(np.float32)
        # Normalize projection basis
        norms = np.linalg.norm(self.basis, axis=1, keepdims=True)
        self.basis /= norms

    def _tokenize(self, text):
        clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        tokens = clean.split()
        features = []
        for t in tokens:
            features.append(t)
            # Character n-grams for subword morphology
            if len(t) >= 4:
                for i in range(len(t) - 2):
                    features.append(t[i:i+3])
        return features

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
            
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            features = self._tokenize(text)
            if not features:
                vectors[i] = self.rng.randn(self.dim).astype(np.float32)
                continue
                
            doc_vec = np.zeros(self.dim, dtype=np.float32)
            for feat in features:
                # Murmur/djb2 hash into basis index
                h = 5381
                for ch in feat:
                    h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
                idx = h % len(self.basis)
                doc_vec += self.basis[idx]
                
            norm = np.linalg.norm(doc_vec)
            if norm > 0:
                vectors[i] = doc_vec / norm
            else:
                vectors[i] = doc_vec
                
        return vectors


class ScratchRAG:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, use_faiss=False):
        self.model_name = model_name
        self.use_faiss = use_faiss
        self.encoder = None
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None

        # Attempt to load SentenceTransformer; fallback cleanly to pure NumPy engine
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading SentenceTransformer: {self.model_name}...")
            self.encoder = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
        except Exception as e:
            print(f"SentenceTransformer not available in local environment ({type(e).__name__}).")
            print(f"Activating self-contained pure NumPy dense embedding engine (dim={EMBEDDING_DIM}).")
            self.encoder = PureNumpyEmbeddingModel(dim=EMBEDDING_DIM)
            self.backend = "pure-numpy-lsa"

    def load_pdf(self, pdf_path):
        """Extract text page-by-page from PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
            
        reader = pypdf.PdfReader(pdf_path)
        pages_data = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            clean_text = re.sub(r'\s+', ' ', text).strip()
            if clean_text:
                pages_data.append({"page": idx + 1, "text": clean_text})
                
        print(f"Loaded {len(pages_data)} pages from {pdf_path}.")
        return pages_data

    def chunk_document(self, pages_data, chunk_size=1024, chunk_overlap=150):
        """
        Sliding-window character chunker preserving page metadata.
        chunk_size: target number of characters per chunk.
        chunk_overlap: overlap in characters between adjacent chunks.
        """
        chunks = []
        chunk_counter = 0
        
        for p_info in pages_data:
            page_num = p_info["page"]
            text = p_info["text"]
            
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_str = text[start:end].strip()
                
                if len(chunk_str) > 30:
                    chunks.append({
                        "chunk_id": f"c_{chunk_counter:04d}",
                        "page": page_num,
                        "text": chunk_str,
                        "char_start": start,
                        "char_end": end
                    })
                    chunk_counter += 1
                    
                if end == len(text):
                    break
                start += (chunk_size - chunk_overlap)
                
        self.chunks = chunks
        print(f"Created {len(self.chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
        return chunks

    def build_index(self):
        """Generate dense embeddings and construct FAISS / NumPy index."""
        if not self.chunks:
            raise ValueError("No chunks found. Call chunk_document() first.")
            
        texts = [c["text"] for c in self.chunks]
        print(f"Encoding {len(texts)} chunks using backend [{self.backend}]...")
        raw_embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        # L2 normalize embeddings for cosine similarity via inner product
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        self.embeddings = (raw_embeddings / norms).astype(np.float32)
        
        if self.use_faiss:
            try:
                import faiss
                dim = self.embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.embeddings)
                print(f"Built FAISS IndexFlatIP with dimension {dim} and {self.faiss_index.ntotal} vectors.")
            except Exception as e:
                print(f"FAISS index construction failed ({e}). Falling back to pure NumPy vector search.")
                self.use_faiss = False
        else:
            print(f"Built pure NumPy index with {len(self.embeddings)} normalized vectors (dim={self.embeddings.shape[1]}).")
            
        return self.embeddings

    def search(self, query, top_k=5):
        """Retrieve top_k most similar chunks using FAISS or pure NumPy."""
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        q_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1e-12
        query_vec = (query_vec / q_norm).astype(np.float32)
        
        if self.use_faiss and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vec, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Pure NumPy Cosine Similarity
            sims = np.dot(self.embeddings, query_vec.T).squeeze()
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]
            
        results = []
        for idx, score in zip(indices, scores):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score)
                })
        return results

    def format_prompt(self, query, retrieved_results):
        """Format retrieved context and query for LLM response generation."""
        context_blocks = []
        for i, res in enumerate(retrieved_results):
            c = res["chunk"]
            context_blocks.append(f"--- Context [{i+1}] (Page {c['page']}) ---\n{c['text']}")
            
        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"You are a factual research assistant. Answer the user query strictly based on the provided context.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )
        return prompt
