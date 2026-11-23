
from typing import Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=5000
    )

    document_id: Optional[str] = None

    top_k: int = Field(
        default=4,
        ge=1,
        le=10
    )
