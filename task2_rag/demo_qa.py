"""
Interactive Demo Script for Task 2: Frameworkless RAG from Scratch.
Demonstrates loading the 86-page NIST AI RMF document, retrieving context with pure NumPy,
and synthesizing factual responses.
"""

import sys
import os
import json
from rag_engine import ScratchRAG

PDF_PATH = "task2_rag/documents/nist_ai_rmf.pdf"
TEST_DATA_PATH = "task2_rag/test_dataset.json"

def main():
    print("=" * 60)
    print("  Frameworkless RAG Engine Demo — NIST AI RMF 1.0 (86 Pages)")
    print("=" * 60)
    
    if not os.path.exists(PDF_PATH) or os.path.getsize(PDF_PATH) < 1000000:
        print("NIST document not found locally. Initiating automated download...")
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "documents"))
        from download_doc import download_document
        download_document()
        
    rag = ScratchRAG(use_faiss=False)
    pages = rag.load_pdf(PDF_PATH)
    rag.chunk_document(pages, chunk_size=1024, chunk_overlap=150)
    rag.build_index()
    
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_questions = json.load(f)
        
    print(f"\nIndexed {len(rag.chunks)} chunks across {len(pages)} pages.")
    print("Testing 3 representative questions from the test set:\n")
    
    sample_indices = [0, 2, 7] # q01 (biases), q03 (aleatoric uncertainty), q08 (assigning bias)
    for idx in sample_indices:
        q_item = test_questions[idx]
        query = q_item["question"]
        print(f"▶ Question: {query}")
        result = rag.answer_query(query, top_k=3)
        print(f"✔ Retrieved Top Source: Page {result['retrieved_chunks'][0]['chunk']['page']} (Score: {result['retrieved_chunks'][0]['score']:.4f})")
        print(f"✔ Extracted Answer: {result['answer']}")
        print("-" * 60)
        
    print("\nDemo finished successfully! Zero frameworks used.")

if __name__ == "__main__":
    main()
