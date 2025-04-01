from typing import List, Optional

from pydantic import BaseModel

"""
This script defines Pydantic models for handling file upload requests, query parameters, and responses in the API.
"""

class UploadFileRequest(BaseModel):
    file_path: str
    collection_name: str
    # length: Optional[int] = 500

class QueryRequest(BaseModel):
    query: str
    collection_name: str
    search_type: Optional[str] = "similarity"
    k: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    context: Optional[List[str]] = None
