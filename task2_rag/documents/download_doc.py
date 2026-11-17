"""
Downloads the NIST AI Risk Management Framework 1.0 (NIST.SP.1270.pdf)
Public domain US Government publication (86 pages).
"""

import os
import urllib.request
import pypdf

DOC_DIR = "task2_rag/documents"
PDF_PATH = os.path.join(DOC_DIR, "nist_ai_rmf.pdf")
URL = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1270.pdf"

def download_document():
    os.makedirs(DOC_DIR, exist_ok=True)
    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1000000:
        print(f"Document already exists at {PDF_PATH} ({os.path.getsize(PDF_PATH)} bytes).")
    else:
        print(f"Downloading NIST AI RMF 1.0 from {URL}...")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response, open(PDF_PATH, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded {len(data)} bytes to {PDF_PATH}.")
        
    reader = pypdf.PdfReader(PDF_PATH)
    print(f"Verified PDF page count: {len(reader.pages)} pages (Requirement: >= 30 pages).")

if __name__ == "__main__":
    download_document()
