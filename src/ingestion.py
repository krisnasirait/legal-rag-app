from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    """
    Loads a PDF and splits it into manageable chunks for the LLM.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of document chunks with metadata
        
    Raises:
        Exception: If PDF loading or splitting fails
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content extracted from PDF")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,     
            chunk_overlap=200,    
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  
        )
        texts = text_splitter.split_documents(documents)
        
        print(f"PDF loaded: {len(documents)} pages, split into {len(texts)} chunks")
        
        return texts
        
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        raise