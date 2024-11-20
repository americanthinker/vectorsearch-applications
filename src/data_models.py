from enum import Enum
from pydantic import BaseModel, ConfigDict

class SearchTypeEnum(str, Enum):
    keyword = 'kw'
    vector = 'vector'
    hybrid = 'hybrid'

class EvaluationDataset(BaseModel):
    """Represents a retrieval dataset for retrieval system evaluation."""

    queries: dict[str, str]
    corpus: dict[str, str]
    relevant_docs: dict[str, str]
    answers: dict[str, str] | None = None

    def __len__(self) -> int:
        return len(self.corpus)
    
class RetrievalEvaluation(BaseModel):
    """Represents the evaluation of a retrieval system."""

    retrieve_limit: int
    top_k: int
    retriever: str
    reranker: str | None = None
    chunk_size: int
    chunk_overlap: int
    total_misses: int
    total_questions: int
    alpha: float | None = None
    model_config = ConfigDict(extra='allow')