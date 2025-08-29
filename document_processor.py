import os
from typing import List, Dict
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    """Handles document processing and chunking for RAG pipeline"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_document(self, file_path: str, file_name: str) -> List[Document]:
        """
        Process a document and return chunks
        
        Args:
            file_path: Path to the document file
            file_name: Original name of the file
            
        Returns:
            List of Document objects with text chunks and metadata
        """
        try:
            # Extract text based on file type
            if file_path.lower().endswith('.pdf'):
                text = self._extract_pdf_text(file_path)
            elif file_path.lower().endswith('.txt'):
                text = self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            if not text.strip():
                raise ValueError(f"No text content found in {file_name}")
            
            # Create document object
            document = Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "file_path": file_path,
                    "file_type": file_path.split('.')[-1].lower(),
                    "char_count": len(text)
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content)
                })
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error processing document {file_name}: {str(e)}")
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        print(f"Warning: Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                raise Exception(f"Error reading text file with multiple encodings: {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading text file: {str(e)}")
    
    def get_document_summary(self, chunks: List[Document]) -> Dict:
        """Get summary statistics for processed document"""
        if not chunks:
            return {}
        
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        
        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "average_chunk_size": avg_chunk_size,
            "source_document": chunks[0].metadata.get("source", "Unknown"),
            "file_type": chunks[0].metadata.get("file_type", "Unknown")
        }
