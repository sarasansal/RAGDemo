from langchain_qdrant import QdrantVectorStore

from scripts.backend.document_processing.document_loader import DocumentLoader
from scripts.backend.document_processing.document_embedder import Embedder
from scripts.backend.document_processing.document_splitter import DocumentSplitter
from scripts.backend.document_processing.vector_store import VectorStore

from scripts.backend.models.rag import UploadFileRequest

import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()


class UploadFile:
    def upload_file(
        self,
        upload_file_request: UploadFileRequest,
        embedder: Embedder,
    ) -> bool:
        try:
            document_loader = DocumentLoader()
            documents = document_loader.load_document(
                file_path=upload_file_request.file_path
            )
            splitter = DocumentSplitter()
            split_documents = splitter.split_text(documents=documents)

            vector_store = VectorStore(
                api_key=os.getenv("QDRANT_API_KEY") , url=os.getenv("QDRANT_URL")
            )
            qdrant_vector_store = vector_store.create_collection(
                collection_name=upload_file_request.collection_name
            )
            qdrant_vector_store = QdrantVectorStore(
                collection_name=upload_file_request.collection_name,
                client=vector_store.client,
                embedding=embedder.embedding
            )
            vector_store.add_documents(
                documents=split_documents,
                qdrant_vector_store=qdrant_vector_store,
            )
            return True
        except Exception as e:
            raise RuntimeError("Error uploading file: " + str(e))
