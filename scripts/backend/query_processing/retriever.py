from typing import List
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from scripts.backend.document_processing.document_embedder import Embedder

class Retriever:
    """
    A class responsible for retrieving relevant documents from a vector store.
    """
    def __init__(
        self, 
        embedder: Embedder, 
        vector_store: QdrantVectorStore
    ):
        """
        Initialize the Retriever.

        Args:
            embedder (Embedder): Embedding model used for query embedding
            vector_store (QdrantVectorStore): Vector store to search in
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve_context(
        self, 
        query: str, 
        collection_name: str, 
        k: int = 5
    ) -> List[Document]:
        """
        Retrieve the most relevant documents for a given query applying similarity search.
        
        Args:
            query (str): User's query
            collection_name (str): Name of the collection to search
            k (int, optional): Number of top similar documents to retrieve. Defaults to 5.
        
        Returns:
            List[Document]: Most relevant documents
        """
        try:
            context_docs = self.vector_store.similarity_search(
                query=query, 
                k=k
            )
            
            return context_docs
        
        except Exception as e:
            raise RuntimeError(f"Error retrieving context: {e}")

    def format_context(
        self, 
        context_docs: List[Document]
    ) -> str:
        """
        Format retrieved documents into a single context string.
        
        Args:
            context_docs (List[Document]): List of retrieved documents
        
        Returns:
            str: Formatted context string
        """
        content_list = []

        for doc in context_docs:
            content = doc.page_content
            content_list.append(content)

        context = "\n\n".join(content_list)

        return context