from fastapi import APIRouter

from scripts.api.routers import rag_routes

api_router = APIRouter()

api_router.include_router(rag_routes.router, prefix="/rag", tags=["rag"])