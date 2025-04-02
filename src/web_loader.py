from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_web_content(url):
    """
    Load web page content using LangChain's WebBaseLoader.

    Args:
        url (str): The URL to load content from.

    Returns:
        list: List of extracted text content.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"❌ Error loading web content: {e}")
        return []

def adjust_chunk_size_for_embedding(text, target_embedding_dim=384, chunk_size=1000, chunk_overlap=200):
    """
    Dynamically adjust chunk size based on the target embedding dimension.

    Args:
        text (str): The text content to be split.
        target_embedding_dim (int): Desired vector size (default is 384).
        chunk_size (int): Initial chunk size (characters).
        chunk_overlap (int): Overlapping characters between chunks.

    Returns:
        list: List of optimized and cleaned text chunks.
    """
    if target_embedding_dim == 384:
        chunk_size = 1000
        chunk_overlap = 100 

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    
    cleaned_chunks = [chunk.replace('\n', ' ').replace('\t', ' ').strip() for chunk in chunks if len(chunk.strip()) > 0]

    return cleaned_chunks
