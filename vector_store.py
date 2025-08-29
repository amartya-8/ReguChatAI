import os
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from openai import OpenAI

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, embedding_dimension: int = 1536):
        """
        Initialize vector store
        
        Args:
            embedding_dimension: Dimension of OpenAI embeddings (1536 for text-embedding-3-small)
        """
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(embedding_dimension)  # Inner product for similarity
        self.documents = []  # Store document metadata
        self.document_count = 0
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Embedding model
        self.embedding_model = "text-embedding-3-small"
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using OpenAI
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = []
            for embedding_obj in response.data:
                embeddings.append(embedding_obj.embedding)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            return
        
        try:
            # Extract text content for embedding
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            embeddings = self._get_embeddings(texts)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store document metadata
            for i, doc in enumerate(documents):
                doc_metadata = doc.metadata.copy()
                doc_metadata.update({
                    'content': doc.page_content,
                    'doc_id': self.document_count + i,
                    'vector_index': len(self.documents) + i
                })
                self.documents.append(doc_metadata)
            
            self.document_count += len(documents)
            
        except Exception as e:
            raise Exception(f"Error adding documents to vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._get_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= score_threshold:  # Valid index and meets threshold
                    doc_metadata = self.documents[idx].copy()
                    doc_metadata['score'] = float(score)
                    results.append(doc_metadata)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        # Count unique documents
        unique_sources = set()
        for doc in self.documents:
            unique_sources.add(doc.get('source', 'Unknown'))
        
        return {
            'total_documents': len(unique_sources),
            'total_chunks': len(self.documents),
            'index_size': self.index.ntotal,
            'embedding_dimension': self.embedding_dimension,
            'sources': list(unique_sources)
        }
    
    def clear(self) -> None:
        """Clear all documents from the vector store"""
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.documents = []
        self.document_count = 0
    
    def get_document_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all chunks from a specific document source"""
        return [doc for doc in self.documents if doc.get('source') == source]
