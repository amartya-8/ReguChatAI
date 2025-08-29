# Regulatory Compliance RAG Chatbot

## Overview

This is a Retrieval-Augmented Generation (RAG) chatbot designed for regulatory compliance professionals. The system allows users to upload regulatory documents (PDF and TXT files) and ask questions to receive context-aware responses based on the document content. The application provides role-based responses tailored for Compliance Analysts and Relationship Managers, ensuring appropriate expertise levels and responsibilities are addressed.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with a conversational chat interface
- **Layout**: Wide layout with expandable sidebar for document management
- **Session Management**: Streamlit session state for maintaining chat history, document loading status, and user roles
- **Caching**: Component-level caching using `@st.cache_resource` for performance optimization

### Backend Architecture
- **Modular Design**: Separated into distinct components for document processing, vector storage, and RAG pipeline
- **Document Processing**: Handles PDF and TXT file extraction with chunking using LangChain's RecursiveCharacterTextSplitter
- **Vector Storage**: FAISS-based vector database for efficient similarity search with OpenAI embeddings
- **RAG Pipeline**: Retrieval-augmented generation using OpenAI's GPT-5 model for context-aware responses

### Core Components
1. **DocumentProcessor**: Extracts and chunks text from uploaded documents with configurable chunk size (1000 characters) and overlap (200 characters)
2. **VectorStore**: Manages document embeddings using FAISS index with OpenAI's text-embedding-3-small model (1536 dimensions)
3. **RAGPipeline**: Orchestrates document retrieval and response generation with role-based customization
4. **Utils**: Provides session state management and chat message formatting utilities

### Data Processing Pipeline
- **Text Extraction**: Supports PDF (PyPDF2) and plain text files
- **Chunking Strategy**: Recursive character splitting with intelligent separators for optimal context preservation
- **Embedding Generation**: OpenAI embeddings for semantic search capabilities
- **Similarity Search**: FAISS inner product similarity for efficient document retrieval

### Response Generation
- **Context Assembly**: Combines retrieved document chunks with user queries
- **Role-Based Prompting**: Customizes responses based on user role (Compliance Analyst vs Relationship Manager)
- **Source Attribution**: Provides relevance scores and source document references for transparency

## External Dependencies

### AI Services
- **OpenAI API**: GPT-5 model for text generation and text-embedding-3-small for document embeddings
- **API Key Management**: Environment variable-based configuration for secure API access

### Core Libraries
- **Streamlit**: Web application framework for user interface
- **FAISS**: Facebook AI Similarity Search for vector database operations
- **LangChain**: Document processing and text splitting utilities
- **PyPDF2**: PDF text extraction capabilities
- **NumPy**: Numerical operations for embedding manipulation

### File Processing
- **Supported Formats**: PDF and TXT files for document upload
- **Temporary Storage**: Uses system temporary files for secure document processing
- **File Size Management**: Includes utilities for human-readable file size formatting

### Environment Configuration
- **OpenAI API Key**: Required environment variable for AI service access
- **Error Handling**: Graceful degradation with user-friendly error messages for missing configurations