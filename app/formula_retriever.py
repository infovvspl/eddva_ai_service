import os
import pdfplumber
import faiss
import numpy as np
import pickle
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

logger = logging.getLogger("ai_services.formula_retriever")

class FormulaRetriever:
    def __init__(self, data_dir: str = "data/knowledge_base", index_path: str = "data/formula_index.faiss"):
        self.data_dir = data_dir
        self.index_path = index_path
        self.metadata_path = index_path.replace(".faiss", ".pkl")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        
        # Load index if exists
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                logger.info(f"Loaded formula index with {len(self.chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to load index: {str(e)}")
                self.index = None

    def index_pdfs(self):
        """Processes all PDFs in data_dir and builds a FAISS index."""
        all_text_chunks = []
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pdf"):
                path = os.path.join(self.data_dir, filename)
                logger.info(f"Indexing PDF: {filename}")
                try:
                    with pdfplumber.open(path) as pdf:
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text()
                            if text:
                                # Split into manageable chunks (e.g., 500 chars)
                                for chunk in [text[i:i+500] for i in range(0, len(text), 400)]:
                                    all_text_chunks.append({
                                        "text": chunk.strip(),
                                        "source": f"{filename} (Page {i+1})"
                                    })
                except Exception as e:
                    logger.error(f"Error reading {filename}: {str(e)}")

        if not all_text_chunks:
            logger.warning("No text found in PDFs to index.")
            return

        # Generate embeddings
        texts = [c["text"] for c in all_text_chunks]
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype('float32')

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(all_text_chunks, f)
            
        self.chunks = all_text_chunks
        logger.info(f"Successfully indexed {len(all_text_chunks)} formula chunks.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Returns relevant formula snippets for a query."""
        if self.index is None or not self.chunks:
            return []

        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.chunks):
                results.append(self.chunks[i])
        return results

# Singleton instance
formula_retriever = FormulaRetriever()
