from enum import Enum
from typing import Optional, Dict, Any, TypedDict

from pydantic import BaseModel, Field


class Intent(str, Enum):
    SUMMARY = "summary"
    BASIC_QA = "basic_qa"
    CODE_GENERATION = "code_generation"
    DEBUG_CODE = "debug_code"
    IMAGE_GENERATION = "image_generation"
    MATH_SOLVE = "math_solve"


class ToolResponseType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    IMAGE = "image"


class IntentAnalysis(BaseModel):
    """Structured output for intent analysis using the mini model."""

    intent: Intent = Field(..., description="Most likely user intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0,1]")
    needs_summary: bool = Field(
        default=False, description="Whether a brief summary should be provided"
    )
    summary: Optional[str] = Field(
        default=None, description="1-3 sentence summary if needs_summary is true"
    )
    reasoning: str = Field(
        ..., description="Short model reasoning for why the intent was chosen"
    )
    input_length: int = Field(..., description="Character length of the input text")


class ToolResponse(BaseModel):
    """Unified tool response model."""

    type: ToolResponseType
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class RouterGraphState(TypedDict, total=False):
    input_text: str
    analysis: IntentAnalysis
    response: ToolResponse


