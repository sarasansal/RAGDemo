from fastapi import APIRouter

from scripts.api.routers import rag_routes

"""
This script aggregates and registers all API routers in the application.
"""

api_router = APIRouter()

api_router.include_router(rag_routes.router, prefix="/rag", tags=["rag"])