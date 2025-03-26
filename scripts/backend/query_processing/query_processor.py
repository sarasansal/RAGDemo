from typing import List
from langchain_qdrant import QdrantVectorStore
from scripts.backend.document_processing.document_embedder import Embedder
from scripts.backend.models.rag import QueryRequest, QueryResponse
from scripts.backend.query_processing.retriever import Retriever
from scripts.backend.query_processing.response_generator import ResponseGenerator

class QueryProcessor:
    """
    A class responsible for processing user queries and generating responses.
    """
    def __init__(
        self, 
        embedder: Embedder, 
        vector_store: QdrantVectorStore,
        response_generator: ResponseGenerator
    ):
        """
        Initialize the QueryProcessor.

        Args:
            embedder (Embedder): Embedding model
            vector_store (QdrantVectorStore): Vector store to search in
            response_generator (ResponseGenerator): Model for generating responses
        """
        self.retriever = Retriever(embedder, vector_store)
        self.response_generator = response_generator

    def process_query(
        self, 
        query_request: QueryRequest
    ) -> QueryResponse:
        """
        Process the query by retrieving context and generating a response.
        
        Args:
            query_request (QueryRequest): Query details
        
        Returns:
            QueryResponse: Generated response based on retrieved context
        """
        try:
            context_docs = self.retriever.retrieve_context(
                query=query_request.query,
                collection_name=query_request.collection_name,
                k=query_request.k
            )
            
            context = self.retriever.format_context(context_docs)
            
            response = self.response_generator.generate_response(
                query=query_request.query, 
                context=context
            )
                        
            return QueryResponse(
                response=response, 
                context=[doc.page_content for doc in context_docs]
            )
        
        except Exception as e:
            return QueryResponse(
                response=f"Error processing query: {str(e)}",
                context=[],
                sources=None
            )