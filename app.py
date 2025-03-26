import warnings  # <-- Esto va PRIMERO
warnings.filterwarnings("ignore", message=".*Tried to instantiate class '__path__._path'.*")

import streamlit as st
from scripts.backend.document_processing.document_embedder import Embedder
from scripts.backend.document_processing.vector_store import VectorStore
from scripts.backend.models.rag import UploadFileRequest, QueryRequest
from scripts.backend.document_processing.upload_file import UploadFile
from scripts.backend.query_processing.query_processor import QueryProcessor
from scripts.backend.query_processing.response_generator import ResponseGenerator

import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def main():
    st.title("RAG Document Q&A System")

    # Sidebar for document upload
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        collection_name = st.text_input("Collection Name", "default_collection")
        
        if uploaded_file and st.button("Upload"):
            # Save uploaded file temporarily
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize components
            embedder = Embedder()
            upload_request = UploadFileRequest(
                file_path=uploaded_file.name, 
                collection_name=collection_name
            )
            
            # Upload file
            upload_service = UploadFile()
            success = upload_service.upload_file(upload_request, embedder)
            
            if success:
                st.success("Document uploaded and indexed successfully!")
            else:
                st.error("Failed to upload document")

    # Main area for querying
    st.header("Ask a Question")
    query = st.text_input("Enter your query:")
    
    if query and collection_name:
        with st.spinner('Generating response...'):
            # Initialize RAG components
            embedder = Embedder()
            vector_store = VectorStore(
                api_key=os.getenv("QDRANT_API_KEY"),
                url=os.getenv("QDRANT_URL")
            )
            
            query_request = QueryRequest(
                query=query,
                collection_name=collection_name,
                k = 5
            )
            
            # Process query and generate response
            response_generator = ResponseGenerator()
            query_processor = QueryProcessor(embedder, vector_store, response_generator)
            
            query_response = query_processor.process_query(query_request)
            
            st.write("### Response:")
            st.write(query_response.response)
            if query_response.context:
                with st.expander("Retrieved Context"):
                    for context in query_response.context:
                        st.write(context)
            else:
                st.warning("No context retrieved.")

if __name__ == "__main__":
    main()