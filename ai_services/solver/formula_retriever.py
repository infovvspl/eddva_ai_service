import os
import numpy as np
import pickle
import logging
from typing import List, Dict, Any

logger = logging.getLogger("ai_services.formula_retriever")

# The formula knowledge base is an OPTIONAL enhancement: it retrieves verified
# formulas from indexed PDFs to ground the scientific solver. Its dependencies
# (sentence-transformers -> torch, faiss, pdfplumber) are heavy and are NOT
# installed in production, so importing them at module level took the whole
# scientific solver down with an ImportError ("No module named
# 'sentence_transformers'") — the doubt endpoint then silently fell back to the
# plain LLM for every science question.
#
# Import them defensively instead. Without them the retriever simply returns no
# formulas and the solver still runs; with them, it enriches the prompt.
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _KB_AVAILABLE = True
    _KB_IMPORT_ERROR = None
except Exception as _exc:  # ImportError, or a broken/partial install
    SentenceTransformer = None
    faiss = None
    _KB_AVAILABLE = False
    _KB_IMPORT_ERROR = _exc

try:
    import pdfplumber  # only needed to (re)build the index, never to query it
except Exception:
    pdfplumber = None


class FormulaRetriever:
    def __init__(self, data_dir: str = "data/knowledge_base", index_path: str = "data/formula_index.faiss"):
        self.data_dir = data_dir
        self.index_path = index_path
        self.metadata_path = index_path.replace(".faiss", ".pkl")
        self.index = None
        self.chunks = []
        self.model = None

        if not _KB_AVAILABLE:
            logger.info(
                "Formula knowledge base disabled (%s). The scientific solver still "
                "runs; it just won't be grounded with retrieved formulas.",
                _KB_IMPORT_ERROR,
            )
            return

        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning("Could not load embedding model (%s) — formula KB disabled.", e)
            self.model = None
            return

        # Load index if exists
        self.load_index()

    @property
    def available(self) -> bool:
        """True when the KB can actually answer queries."""
        return bool(self.model is not None and self.index is not None)

    def load_index(self):
        if not _KB_AVAILABLE or self.model is None:
            return
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
        if not _KB_AVAILABLE or self.model is None or pdfplumber is None:
            raise RuntimeError(
                "Cannot build the formula index: the knowledge-base extras are not "
                "installed (need sentence-transformers, faiss, pdfplumber). "
                f"Import error: {_KB_IMPORT_ERROR}"
            )

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
        """
        Return relevant formula snippets for a query.

        Returns [] when the knowledge base is unavailable (extras not installed,
        or no index built) — the solver then runs without formula grounding
        rather than failing.
        """
        if not self.available or not self.chunks:
            return []

        try:
            query_vector = self.model.encode([query]).astype('float32')
            distances, indices = self.index.search(query_vector, top_k)
        except Exception as e:
            logger.warning("Formula retrieval failed (%s) — continuing without it.", e)
            return []

        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.chunks):
                results.append(self.chunks[i])
        return results

# Singleton instance
formula_retriever = FormulaRetriever()
