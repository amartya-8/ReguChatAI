import streamlit as st
from typing import Dict, Any

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'user_role' not in st.session_state:
        st.session_state.user_role = "Compliance Analyst"

def display_chat_message(message: Dict[str, Any]):
    """
    Display a chat message in the Streamlit interface
    
    Args:
        message: Dictionary containing message data
    """
    role = message.get("role", "user")
    content = message.get("content", "")
    sources = message.get("sources", [])
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Display sources if this is an assistant message with sources
        if role == "assistant" and sources:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:** {source['document']}")
                    st.markdown(f"*Relevance Score: {source['score']:.2f}*")
                    st.markdown(f"```\n{source['content'][:300]}...\n```")
                    if i < len(sources):
                        st.markdown("---")

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes = int(size_bytes / 1024.0)
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_api_key() -> bool:
    """
    Validate if OpenAI API key is available
    
    Returns:
        True if API key is available, False otherwise
    """
    import os
    api_key = os.getenv("OPENAI_API_KEY", "")
    return bool(api_key and api_key != "your-openai-api-key-here")

def display_error_message(error: str, context: str = ""):
    """
    Display a formatted error message
    
    Args:
        error: Error message
        context: Additional context about the error
    """
    st.error(f"**Error:** {error}")
    if context:
        st.error(f"**Context:** {context}")

def display_success_message(message: str):
    """
    Display a formatted success message
    
    Args:
        message: Success message to display
    """
    st.success(f"✅ {message}")

def display_warning_message(message: str):
    """
    Display a formatted warning message
    
    Args:
        message: Warning message to display
    """
    st.warning(f"⚠️ {message}")

def get_role_description(role: str) -> str:
    """
    Get description for a user role
    
    Args:
        role: User role
        
    Returns:
        Role description
    """
    descriptions = {
        "Compliance Analyst": "Receives detailed technical responses with regulatory references and procedural information",
        "Relationship Manager": "Receives business-focused responses in plain language with practical implications"
    }
    return descriptions.get(role, "Standard user role")

def format_chunk_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format chunk metadata for display
    
    Args:
        metadata: Chunk metadata dictionary
        
    Returns:
        Formatted metadata string
    """
    source = metadata.get('source', 'Unknown')
    chunk_id = metadata.get('chunk_id', 0)
    total_chunks = metadata.get('total_chunks', 0)
    file_type = metadata.get('file_type', 'unknown')
    
    return f"{source} (Chunk {chunk_id + 1}/{total_chunks}, {file_type.upper()})"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def get_file_type_icon(file_type: str) -> str:
    """
    Get icon for file type
    
    Args:
        file_type: File type extension
        
    Returns:
        Unicode icon for file type
    """
    icons = {
        'pdf': '📄',
        'txt': '📝',
        'doc': '📄',
        'docx': '📄'
    }
    return icons.get(file_type.lower(), '📋')
