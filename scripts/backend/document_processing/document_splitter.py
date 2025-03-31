from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentSplitter:
    """
    A class to split text into chunks with the RecursiveCharacterTextSplitter method.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the DocumentSplitter.

        Args:
            chunk_size (int, optional): Maximum number of characters in each chunk. 
                Defaults to 500.
            chunk_overlap (int, optional): Number of characters to overlap between chunks. 
                Defaults to 50.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Splits text into chunks and returns a list of Documents.

        Args:
            documents (List[Document]): List of documents to split.

        Returns:
            List[Document]: List of document chunks.
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'splitter': 'RecursiveCharacterTextSplitter',
                    'chunk_size': len(chunk.page_content),
                })

        except Exception as e:
            raise RuntimeError(f"Failed to split documents: {e}")
        
        return chunks
