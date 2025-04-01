from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from scripts.api.main import api_router
from scripts.backend.document_processing.document_embedder import Embedder

"""  
This script sets up a FastAPI application with lifecycle management, custom route IDs, CORS support, and an API router.

1. Defines the lifecycle, initializing an Embedder when the app starts.  
2. Creates a function for unique route IDs based on tags and names.  
3. Initializes the FastAPI app with custom settings and lifecycle management.  
4. Adds CORS middleware to allow unrestricted access.  
5. Includes the API router under '/api/v1'.  
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedder = Embedder(model_name="bert-large-uncased")
    yield

def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"

app = FastAPI(
    title="My BabyRAG Project",
    openapi_url="/openapi.json",  
    docs_url="/docs",            
    redoc_url="/redoc",          
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")