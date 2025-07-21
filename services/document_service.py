"""Document processing and vector store operations."""
import os
import glob
import logging
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize components
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])


def load_documents(path: str) -> List:
    """Load documents from file or directory path."""
    documents = []
    
    if os.path.isfile(path):
        documents.extend(_load_single_file(path))
    elif os.path.isdir(path):
        for ext in ['*.txt', '*.pdf', '*.docx']:
            files = glob.glob(os.path.join(path, '**', ext), recursive=True)
            for file in files:
                documents.extend(_load_single_file(file))
    
    return documents


def _load_single_file(file_path: str) -> List:
    """Load a single file based on extension."""
    try:
        loaders = {
            '.txt': lambda: TextLoader(file_path, encoding='utf-8'),
            '.pdf': lambda: PyPDFLoader(file_path),
            '.docx': lambda: Docx2txtLoader(file_path)
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in loaders:
            logger.warning(f"Unsupported file type: {ext}")
            return []
        
        documents = loaders[ext]().load()
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def create_vector_store(documents: List) -> Optional[PineconeVectorStore]:
    """Create vector store from documents."""
    if not documents:
        return None
    
    try:
        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        
        # Setup Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = os.environ["INDEX_NAME"]
        
        # Ensure index exists with correct dimension
        _ensure_index_exists(pc, index_name)
        
        # Create vector store
        vectorstore = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
        logger.info(f"Created vector store with {len(chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None


def get_vector_store() -> Optional[PineconeVectorStore]:
    """Get existing vector store."""
    try:
        return PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
    except Exception as e:
        logger.error(f"Error getting vector store: {e}")
        return None


def delete_all_documents() -> bool:
    """Delete all documents from Pinecone index."""
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["INDEX_NAME"])
        index.delete(delete_all=True)
        return True
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        return False


def _ensure_index_exists(pc: Pinecone, index_name: str) -> None:
    """Ensure Pinecone index exists with correct configuration."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
        )
        import time
        time.sleep(10)  # Wait for initialization
    else:
        # Check dimension compatibility
        index_info = pc.describe_index(index_name)
        if index_info.dimension != 1536:
            pc.delete_index(index_name)
            import time
            time.sleep(5)
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            time.sleep(10)
