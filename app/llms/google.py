from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import ToolResponse
from schemas.enums import ToolResponseType 


def call_gemini_image_model(text: str) -> ToolResponse:
    """Call the Gemini API."""
    model_name = "gemini-nano-banana"
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    image_result = model.invoke(text)
    # TODO : Upload image to storage and return url
    return ToolResponse(type=ToolResponseType.IMAGE, content=image_result.content)

def call_gemini(model_name: str, system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Gemini API."""
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])