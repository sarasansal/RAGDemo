from typing import Optional

from pydantic import BaseModel


class UploadFileRequest(BaseModel):
    file_path: str
    collection_name: str
    length: Optional[int] = 500


# class QueryRequest(BaseModel):
#     query: str
#     collection_name: str
#     search_type: Optional[str] = "similarity"
#     k: Optional[int] = 150
#     use_web_retrieval: Optional[bool] = False
#     use_reranking: Optional[bool] = True
#     use_hyde_transformation: Optional[bool] = False
#     use_multi_step_query: Optional[bool] = False
#     on_premise: Optional[bool] = False


# class QueryResponse(BaseModel):
#     response: str
