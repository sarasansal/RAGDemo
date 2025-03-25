from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentSplitter:
    """
    A class to split text into chunks with the RecursiveCharacterTextSplitter method.

    Attributes:
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the DocumentSplitter.

        Args:
            chunk_size (int, optional): Maximum number of characters in each chunk. 
                Defaults to 1000.
            chunk_overlap (int, optional): Number of characters to overlap between chunks. 
                Defaults to 200.
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
            
            # Add a chunk index
            # for i, chunk in enumerate(chunks):
            #     chunk.metadata['chunk_id'] = i

        except Exception as e:
            raise RuntimeError(f"Failed to split documents: {e}")
        
        return chunks
