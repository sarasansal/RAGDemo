import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

class DocumentLoader:
    """
    A class to load PDF documents.

    Attributes:
        None
    """

    def __init__(
        self
    ):
        pass

    def load_document(
        self, file_path: str = ""
    ) -> List[Document]:
        """
        Load a PDF document.

        Args:
            file_path (str): Full path to the PDF file.

        Returns:
            List[Document]: List of loaded documents.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a PDF.
            RuntimeError: If there is an error while loading the PDF.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() != '.pdf':
            raise ValueError(f"File {file_path} is not a PDF")
        
        loader = PyPDFLoader(file_path)
        try:
            documents = loader.load()
        except Exception as e:
            raise RuntimeError(f"Error cargando {file_path}: {e}") from e
        
        for doc in documents:
            doc.metadata.update({
                'source': os.path.basename(file_path),
                'file_path': file_path,
                'file_type': 'pdf',
                'loader': 'PyPDFLoader'
            })
        
        return documents



