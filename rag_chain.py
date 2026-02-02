"""
TechGear Electronics - RAG Chain Implementation
This module provides the RAG (Retrieval Augmented Generation) chain
using Google Gemini and ChromaDB for context-aware responses.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "techgear_products"

# System prompt for the support agent
SYSTEM_PROMPT = """You are a helpful and friendly customer support agent for TechGear Electronics.
Your role is to assist customers with their product inquiries, technical support questions, and general information.

Guidelines:
- Be polite, professional, and helpful at all times
- Provide accurate information based on the context provided
- If the information is not in the context, say you don't have that specific information and suggest contacting support
- Include relevant product details like prices, features, and SKUs when available
- For troubleshooting, provide clear step-by-step instructions
- Keep responses concise but comprehensive

Context from knowledge base:
{context}

Customer Query: {question}

Response:"""


class RAGChain:
    """RAG Chain for TechGear Electronics customer support."""
    
    def __init__(self, chroma_path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        """Initialize the RAG chain with ChromaDB and Gemini."""
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self._vectorstore = None
        self._retriever = None
        self._chain = None
        self._llm = None
        
    def _get_embeddings(self):
        """Get the embedding model."""
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    def _get_llm(self):
        """Get the Gemini LLM."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3,
                max_tokens=1024,
            )
        return self._llm
    
    def get_vectorstore(self):
        """Get or create the ChromaDB vector store."""
        if self._vectorstore is None:
            embeddings = self._get_embeddings()
            self._vectorstore = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=embeddings,
                collection_name=self.collection_name
            )
        return self._vectorstore
    
    def get_retriever(self, k: int = 4):
        """Get the retriever from the vector store."""
        if self._retriever is None:
            vectorstore = self.get_vectorstore()
            self._retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        return self._retriever
    
    def _format_docs(self, docs) -> str:
        """Format retrieved documents into a single context string."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    def get_chain(self):
        """Build and return the RAG chain."""
        if self._chain is None:
            retriever = self.get_retriever()
            llm = self._get_llm()
            
            prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
            
            self._chain = (
                {
                    "context": retriever | self._format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
        return self._chain
    
    def invoke(self, query: str) -> str:
        """
        Process a query through the RAG chain.
        
        Args:
            query: The user's question or inquiry
            
        Returns:
            The generated response from the RAG chain
        """
        chain = self.get_chain()
        response = chain.invoke(query)
        return response
    
    def retrieve_context(self, query: str, k: int = 4) -> list:
        """
        Retrieve relevant context without generating a response.
        Useful for debugging or inspecting retrieved documents.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        retriever = self.get_retriever(k=k)
        return retriever.invoke(query)


# Create a singleton instance for use across the application
_rag_chain_instance: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get or create the RAG chain singleton instance."""
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = RAGChain()
    return _rag_chain_instance


def query_rag(question: str) -> str:
    """
    Convenience function to query the RAG chain.
    
    Args:
        question: The user's question
        
    Returns:
        The generated response
    """
    rag = get_rag_chain()
    return rag.invoke(question)


# For testing
if __name__ == "__main__":
    print("Testing RAG Chain...")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "What is the price of SmartWatch Pro X?",
        "How do I reset my wireless earbuds?",
        "What is your return policy?",
    ]
    
    rag = get_rag_chain()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        try:
            response = rag.invoke(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print()
