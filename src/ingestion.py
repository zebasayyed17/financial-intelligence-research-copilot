from pathlib import Path
import fitz  # PyMuPDF

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """Split text into overlapping chunks for finer retrieval granularity."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

def ingest_pdfs(reports_dir: Path):
    """Extract text page-by-page from all PDFs in the reports directory."""
    chunks = []
    pdf_files = list(reports_dir.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if not text or not text.strip():
                    continue
                # Chunk individual pages
                page_chunks = chunk_text(text)
                for chunk in page_chunks:
                    chunks.append({
                        "source": pdf_path.name,
                        "page": page_num + 1,
                        "text": chunk
                    })
        except Exception as e:
            print(f"Error reading {pdf_path.name}: {e}")
            
    return chunks