from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class VectorStore:
    """
    A class that manages a Qdrant connection, enabling collection creation and document storage in a vector database.
    """

    def __init__(
        self,
        api_key: str,
        url: str
    ):
        """
        Initialize a VectorStore.

        Args:
            api_key (str): The API key for Qdrant.
            url (str): The URL of the Qdrant server.
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.api_key = api_key
        self.url = url

    def create_collection(
        self,
        collection_name: str,
        size: int = 1024,
        distance: Distance = Distance.COSINE,
    )-> bool:
        """
        Create a collection in Qdrant.

        Args:
            collection_name (str): The name of the collection.
            size (int): The size of the vectors. Defaults to 1024.
            distance (Distance): The distance metric to use. Defaults to COSINE.

        Returns:
            bool: True if collection is created, False if it already exists.
        """
        if self.client.collection_exists(collection_name=collection_name):
            return False
            
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )
        return True

    def add_documents(
        self,
        documents: List[Document],
        qdrant_vector_store: QdrantVectorStore,
    ):
        """
        Add documents to the collection.

        Args:
            documents (List[Document]): The documents to add.
            qdrant_vector_store (QdrantVectorStore): The QdrantVectorStore to use.
        """
        for document in documents:
            qdrant_vector_store.add_documents(documents=[document])
