from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
from src.llm.prompt_templates import huberman_system_message, question_answering_user_message

class SearchTypeEnum(str, Enum):
    keyword = 'kw'
    vector = 'vector'
    hybrid = 'hybrid'

class PromptMessages(str, Enum):
    system = huberman_system_message
    user = question_answering_user_message

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

class GenerationEvaluation(BaseModel):
    """Represents the evaluation of an answer generation system."""

    eval_model: str
    reader_model: str
    temperature: float
    num_context_chunks: int
    verbosity: str | int
    context_chunk_size: int
    context_chunk_overlap: int
    small_to_big: bool
    system_message: str
    user_message: str
    total_questions: int | None = Field(default=None)
    correct_answers: int | None = Field(default=None)
    incorrect_answers: int | None = Field(default=None)
    eval_score: float | None = Field(default=None)