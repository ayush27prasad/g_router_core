from typing import Optional, List

from openai import OpenAI
from langchain_core.messages import BaseMessage
from app.system_prompts import DEFAULT_PROMPT

def call_onboarded_model(base_url: str, api_key: str, model_name: str, user_query: str, messages: Optional[List[BaseMessage]] = None) -> str:
    """Call an OpenAI-compatible API. If messages are provided, use them as chat history."""

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    if messages and len(messages) > 0:
        converted = []
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None)
            content = getattr(m, "content", None)
            if role is None and hasattr(m, "__class__"):
                clsname = m.__class__.__name__.lower()
                role = "assistant" if "ai" in clsname else "user"
            if role in ("human", "user"):
                role = "user"
            elif role in ("ai", "assistant"):
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                role = "user"
            converted.append({"role": role, "content": content})
        payload_messages = [{"role": "system", "content": DEFAULT_PROMPT}] + converted
    else:
        payload_messages = [
            {"role": "system", "content": DEFAULT_PROMPT},
            {"role": "user", "content": user_query}
        ]

    resp = client.chat.completions.create(model=model_name, messages=payload_messages)
    response_content = resp.choices[0].message.content
    print(f"{model_name} response : {response_content}")
    return response_content