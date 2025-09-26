from typing import TypedDict, Optional

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from .enums import Intent, ToolResponseType


class IntentAnalysis(BaseModel):
    """Structured output for intent analysis using the mini model."""
    intent: Intent = Field(..., description="Most likely user intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0,1]")
    reasoning: str = Field(..., description="Short model reasoning for why the intent was chosen.")

class ToolResponse(BaseModel):
    """Unified tool response model."""
    type: ToolResponseType
    content: str
    response_generated_via: Optional[str]

class RouterGraphState(TypedDict, total=False):
    input_text: str
    messages: Annotated[list, add_messages]
    request_model_name: Optional[str]
    response_model_name: str
    analysis: IntentAnalysis
    response: ToolResponse


