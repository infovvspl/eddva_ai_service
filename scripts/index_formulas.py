import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.formula_retriever import formula_retriever

def main():
    print("Starting formula indexing process...")
    try:
        formula_retriever.index_pdfs()
        print("Indexing completed successfully!")
        print(f"Index saved at: {formula_retriever.index_path}")
    except Exception as e:
        print(f"Error during indexing: {str(e)}")

if __name__ == "__main__":
    main()
