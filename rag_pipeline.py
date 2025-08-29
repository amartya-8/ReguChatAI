import os
from typing import Dict, List, Any
from openai import OpenAI
from vector_store import VectorStore

class RAGPipeline:
    """RAG (Retrieval Augmented Generation) pipeline for compliance Q&A"""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Vector store instance for document retrieval
        """
        self.vector_store = vector_store
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        self.openai_client = OpenAI(api_key=api_key)
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-5"
    
    def get_response(self, query: str, user_role: str, k: int = 5) -> Dict[str, Any]:
        """
        Get RAG response for a query
        
        Args:
            query: User question
            user_role: User's role (Compliance Analyst or Relationship Manager)
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer and sources
        """
        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find relevant information in the uploaded documents to answer your question. Please ensure you've uploaded the appropriate regulatory documents or try rephrasing your question.",
                    'sources': []
                }
            
            # Step 2: Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)
            
            # Step 3: Generate role-specific prompt
            prompt = self._create_role_specific_prompt(query, context, user_role)
            
            # Step 4: Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(user_role)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Step 5: Format sources
            sources = self._format_sources(relevant_docs)
            
            return {
                'answer': answer,
                'sources': sources,
                'context_used': len(relevant_docs)
            }
            
        except Exception as e:
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}. Please try again or contact support if the issue persists.",
                'sources': []
            }
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.get('source', 'Unknown Document')
            content = doc.get('content', '')
            score = doc.get('score', 0)
            
            context_parts.append(
                f"[Source {i}: {source} (Relevance: {score:.2f})]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _get_system_prompt(self, user_role: str) -> str:
        """Get role-specific system prompt"""
        base_prompt = """You are an expert regulatory compliance assistant. Your role is to provide accurate, helpful, and context-aware responses based on the regulatory documents provided. Always cite your sources and be transparent about the limitations of your knowledge."""
        
        role_specific_prompts = {
            "Compliance Analyst": """
You are responding to a Compliance Analyst. Provide detailed, technical responses with:
- Specific regulatory references and citations
- Technical compliance terminology
- Detailed procedural information
- Risk assessment considerations
- Cross-references to related regulations when applicable
""",
            "Relationship Manager": """
You are responding to a Relationship Manager. Provide clear, business-focused responses with:
- Plain language explanations of regulatory requirements
- Practical implications for client relationships
- Key takeaways and action items
- Customer-facing guidance
- Focus on business impact rather than technical details
"""
        }
        
        return base_prompt + role_specific_prompts.get(user_role, "")
    
    def _create_role_specific_prompt(self, query: str, context: str, user_role: str) -> str:
        """Create a role-specific prompt for the LLM"""
        prompt = f"""
Based on the following regulatory documents and context, please answer the user's question.

CONTEXT FROM REGULATORY DOCUMENTS:
{context}

USER QUESTION: {query}

RESPONSE REQUIREMENTS:
1. Provide a comprehensive answer based on the provided context
2. Always cite specific sources from the context when making claims
3. If the context doesn't fully answer the question, clearly state what information is missing
4. Tailor your response to a {user_role} audience
5. Include relevant warnings or disclaimers about regulatory compliance
6. If applicable, mention related regulations or considerations

Please provide your response:
"""
        return prompt
    
    def _format_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for display"""
        sources = []
        
        for doc in documents:
            source_info = {
                'document': doc.get('source', 'Unknown Document'),
                'content': doc.get('content', '')[:500] + '...' if len(doc.get('content', '')) > 500 else doc.get('content', ''),
                'score': doc.get('score', 0),
                'chunk_id': doc.get('chunk_id', 0),
                'file_type': doc.get('file_type', 'unknown')
            }
            sources.append(source_info)
        
        return sources
    
    def get_document_summary(self, source: str) -> str:
        """Get a summary of a specific document"""
        try:
            doc_chunks = self.vector_store.get_document_by_source(source)
            
            if not doc_chunks:
                return f"No document found with source: {source}"
            
            # Combine all chunks from the document
            full_content = "\n".join([chunk.get('content', '') for chunk in doc_chunks])
            
            prompt = f"""
Please provide a comprehensive summary of the following regulatory document:

DOCUMENT: {source}
CONTENT:
{full_content[:4000]}  # Limit content to avoid token limits

Please provide a structured summary including:
1. Document purpose and scope
2. Key regulatory requirements
3. Important deadlines or timelines
4. Compliance obligations
5. Penalties or consequences for non-compliance
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a regulatory compliance expert. Provide clear, structured summaries of regulatory documents."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            return response.choices[0].message.content or "No summary generated"
            
        except Exception as e:
            return f"Error generating document summary: {str(e)}"
