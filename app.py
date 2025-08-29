import streamlit as st
import os
import tempfile
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from utils import initialize_session_state, display_chat_message

# Page configuration
st.set_page_config(
    page_title="Regulatory Compliance RAG Chatbot",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

# Main title
st.title("📋 Regulatory Compliance RAG Chatbot")
st.markdown("Upload regulatory documents and ask questions to get instant, context-aware responses.")

# Initialize components
@st.cache_resource
def get_components():
    """Initialize and cache the main components"""
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    rag_pipeline = RAGPipeline(vector_store)
    return doc_processor, vector_store, rag_pipeline

try:
    doc_processor, vector_store, rag_pipeline = get_components()
except Exception as e:
    st.error(f"Failed to initialize components: {str(e)}")
    st.error("Please ensure your OpenAI API key is properly configured.")
    st.stop()

# Sidebar for document management
with st.sidebar:
    st.header("📄 Document Management")
    
    # Role selection
    st.subheader("👤 Select Your Role")
    role = st.selectbox(
        "Choose your role for customized responses:",
        ["Compliance Analyst", "Relationship Manager"],
        help="Different roles receive responses tailored to their expertise level and responsibilities."
    )
    st.session_state.user_role = role
    
    # Document upload
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload regulatory documents (PDF, TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload regulatory documents like AML, KYC, GDPR policies, etc."
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Process document
                        chunks = doc_processor.process_document(tmp_file_path, uploaded_file.name)
                        
                        # Add to vector store
                        vector_store.add_documents(chunks)
                        
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
                st.session_state.documents_loaded = True
    
    # Document statistics
    if st.session_state.documents_loaded:
        st.subheader("📊 Document Statistics")
        stats = vector_store.get_statistics()
        st.metric("Total Documents", stats['total_documents'])
        st.metric("Total Chunks", stats['total_chunks'])
        
        if st.button("Clear All Documents", type="secondary"):
            vector_store.clear()
            st.session_state.documents_loaded = False
            st.session_state.chat_history = []
            st.success("All documents cleared!")
            st.rerun()

# Main chat interface
st.header("💬 Ask Questions")

# Display chat history
for message in st.session_state.chat_history:
    display_chat_message(message)

# Chat input
if prompt := st.chat_input("Ask about regulatory compliance..."):
    if not st.session_state.documents_loaded:
        st.warning("Please upload and process documents first before asking questions.")
    else:
        # Add user message to chat history
        user_message = {"role": "user", "content": prompt}
        st.session_state.chat_history.append(user_message)
        display_chat_message(user_message)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                try:
                    response = rag_pipeline.get_response(prompt, st.session_state.user_role)
                    
                    # Display main response
                    st.markdown(response['answer'])
                    
                    # Display sources
                    if response['sources']:
                        with st.expander("📚 Sources", expanded=False):
                            for i, source in enumerate(response['sources'], 1):
                                st.markdown(f"**Source {i}:** {source['document']}")
                                st.markdown(f"*Relevance Score: {source['score']:.2f}*")
                                st.markdown(f"```\n{source['content'][:300]}...\n```")
                                st.markdown("---")
                    
                    # Add assistant response to chat history
                    assistant_message = {
                        "role": "assistant", 
                        "content": response['answer'],
                        "sources": response['sources']
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    "**Note:** This chatbot provides information based on uploaded regulatory documents. "
    "Always verify critical compliance information with official sources and legal counsel."
)
