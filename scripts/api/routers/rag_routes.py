
import os
from dotenv import load_dotenv

from fastapi import APIRouter, HTTPException

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from scripts.api.dependencies import EmbedderDepends
from scripts.backend.query_processing.query_processor import QueryProcessor
from scripts.backend.document_processing.upload_file import UploadFile
from scripts.backend.models.rag import UploadFileRequest, QueryRequest, QueryResponse
from scripts.backend.query_processing.response_generator import ResponseGenerator

"""
This script defines API endpoints for uploading files and querying a vector database using Qdrant.
"""

load_dotenv()

router = APIRouter()

@router.post("/upload")
def upload_file(
    upload_file_request: UploadFileRequest,
    embedder: EmbedderDepends
):
    try:
        UploadFile().upload_file(upload_file_request, embedder)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/query", response_model=None)
def query(
    query_request: QueryRequest, embedder: EmbedderDepends
):
    try:
        qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
            
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=query_request.collection_name,
            embedding=embedder.embedding
        )
            
        query_request = QueryRequest(
            query=query_request.query,
            collection_name=query_request.collection_name,
            k=5
        )
            
        response_generator = ResponseGenerator()
        query_processor = QueryProcessor(embedder, vector_store, response_generator)
            
        query_response = query_processor.process_query(query_request)
        response = query_response.response
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))