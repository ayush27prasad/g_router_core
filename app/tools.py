from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from llms.openai import get_mini_model
from models.enums import ToolResponseType
from models.models import IntentAnalysis, ToolResponse

from system_prompts import INTENT_CLASSIFIER_PROMPT

def analyze_intent(user_query: str) -> IntentAnalysis:
    """Analyze user intent only using structured output."""
    classifier_model = get_mini_model().with_structured_output(schema=IntentAnalysis)
    return classifier_model.invoke([SystemMessage(INTENT_CLASSIFIER_PROMPT), HumanMessage(user_query)])

