from typing import Annotated

from fastapi import Depends, Request
from scripts.backend.document_processing.document_embedder import Embedder

"""
This script defines a FastAPI dependency to retrieve the 'Embedder' object from the app's state,
ensuring it is initialized before being used in the application routes.
The 'EmbedderDepends' is an alias for this dependency.
"""

def get_embedder(request: Request) -> Embedder:
    embedder = request.app.state.embedder
    if embedder is None:
        raise RuntimeError("Embedder is not initialized")
    return embedder


EmbedderDepends = Annotated[Embedder, Depends(get_embedder)]