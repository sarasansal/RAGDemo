from typing import Annotated

from fastapi import Depends, Request
from scripts.backend.document_processing.document_embedder import Embedder

def get_embedder(request: Request) -> Embedder:
    embedder = request.app.state.embedder
    if embedder is None:
        raise RuntimeError("Embedder is not initialized")
    return embedder


EmbedderDepends = Annotated[Embedder, Depends(get_embedder)]