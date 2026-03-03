from pydantic import BaseModel
from typing import Any, Dict

class QueryRequest(BaseModel):
    question: str

class ClassificationResult(BaseModel):
    domain: str
    confidence: float

class AgentResult(BaseModel):
    domain: str
    answer: str
    metadata: Dict[str, Any] | None = None

class QueryResponse(BaseModel):
    classification: ClassificationResult
    result: AgentResult