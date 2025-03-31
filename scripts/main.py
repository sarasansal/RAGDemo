from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from scripts.api.main import api_router
from scripts.backend.document_processing.document_embedder import Embedder

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cambié a un modelo más ligero para desarrollo
    app.state.embedder = Embedder(model_name="bert-large-uncased")  # Modelo más rápido
    yield

def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"

app = FastAPI(
    title="My BabyRAG Project",
    # Cambios críticos aquí:
    openapi_url="/openapi.json",  # Ruta directa sin prefijo
    docs_url="/docs",            # Habilita docs en la raíz
    redoc_url="/redoc",          # Habilita redoc en la raíz
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montaje del router con prefijo
app.include_router(api_router, prefix="/api/v1")