from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI

from models import ToolResponse, ToolResponseType 


def call_gemini_image_model(text: str) -> ToolResponse:
    """Call the Gemini API."""
    model_name = "gemini-nano-banana"
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    image_result = model.invoke(text)
    # TODO : Upload image to storage and return url
    return ToolResponse(type=ToolResponseType.IMAGE, content=image_result.content, meta={"tool": "gemini_image_model"})