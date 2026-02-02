"""
TechGear Electronics - Knowledge Base Ingestion Script
This script loads the product knowledge base, splits it into chunks,
creates embeddings, and stores them in ChromaDB.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = "data/product_info.txt"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "techgear_products"


def load_documents(file_path: str):
    """Load documents from a text file."""
    print(f"Loading documents from {file_path}...")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into smaller chunks for better retrieval."""
    print(f"Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks


def create_embeddings():
    """Create embedding function using Google Generative AI."""
    print("Initializing Google Generative AI Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return embeddings


def store_in_chroma(chunks, embeddings, persist_directory: str, collection_name: str):
    """Store document chunks in ChromaDB."""
    print(f"Storing {len(chunks)} chunks in ChromaDB at {persist_directory}...")
    
    # Create Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB.")
    return vectorstore


def main():
    """Main function to run the ingestion pipeline."""
    print("=" * 60)
    print("TechGear Electronics - Knowledge Base Ingestion")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key:")
        print("  GOOGLE_API_KEY=your_api_key_here")
        return
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        return
    
    # Run ingestion pipeline
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = store_in_chroma(chunks, embeddings, CHROMA_PATH, COLLECTION_NAME)
    
    print("=" * 60)
    print("Ingestion complete!")
    print(f"ChromaDB stored at: {CHROMA_PATH}")
    print("=" * 60)
    
    # Test retrieval
    print("\nTesting retrieval with sample query...")
    results = vectorstore.similarity_search("What is the price of SmartWatch Pro X?", k=2)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200] + "...")


if __name__ == "__main__":
    main()
