from typing import TypedDict, Annotated

from pydantic import BaseModel, Field

from .enums import Intent, ToolResponseType, ModelName


class IntentAnalysis(BaseModel):
    """Structured output for intent analysis using the mini model."""
    intent: Intent = Field(..., description="Most likely user intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0,1]")
    reasoning: str = Field(..., description="Short model reasoning for why the intent was chosen.")

class ToolResponse(BaseModel):
    """Unified tool response model."""
    type: ToolResponseType
    content: str
    response_generated_via: str

class RouterGraphState(TypedDict, total=False):
    input_text: str
    request_model_name: ModelName
    response_model_name: ModelName
    analysis: IntentAnalysis
    response: ToolResponse


