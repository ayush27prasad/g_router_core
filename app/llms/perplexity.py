from typing import Optional, List

from langchain_perplexity import ChatPerplexity
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from app.schemas.models import ToolResponse
from app.schemas.enums import ToolResponseType
from openai import OpenAI
import os


def call_perplexity(model_name: str, system_msg: str, human_msg: str, messages: Optional[List[BaseMessage]] = None) -> ToolResponse:
    """Call the Perplexity API, optionally with chat history."""
    model = ChatPerplexity(model=model_name, temperature=0.3)
    if messages and len(messages) > 0:
        full_messages: List[BaseMessage] = [SystemMessage(system_msg)] + list(messages)
        response = model.invoke(full_messages)
    else:
        response = model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])
    return ToolResponse(type=ToolResponseType.TEXT, content=str(response))

def call_perplexity_api(model_name: str, system_msg: str, human_msg: str, messages: Optional[List[BaseMessage]] = None) -> str:
    """Call the Perplexity API directly."""
    API_KEY = os.getenv("PPLX_API_KEY")

    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.perplexity.ai"
    )

    if messages and len(messages) > 0:
        converted = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role is None and hasattr(m, "__class__"):
                clsname = m.__class__.__name__.lower()
                role = "assistant" if "ai" in clsname else "user"
            # Normalize roles to OpenAI-style
            if role in ("human", "user"):
                role = "user"
            elif role in ("ai", "assistant"):
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                role = "user"
            converted.append({"role": role, "content": content})
        payload_messages = [{"role": "system", "content": system_msg}] + converted
    else:
        payload_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": human_msg},
        ]

    resp = client.chat.completions.create(model=model_name, messages=payload_messages)
    response_content = resp.choices[0].message.content
    print(f"Perplexity Response : {response_content}")
    return response_content







