import os
from typing import Optional, List

from sarvamai import SarvamAI

from app.schemas.models import ToolResponse
from app.schemas.enums import ToolResponseType
from app.system_prompts import SARVAM_PROMPT


VALID_ROLES = {"system", "user", "assistant"}

def call_sarvam(user_query: str, messages: Optional[List] = None) -> ToolResponse:
    """Call the Sarvam API, optionally with chat history."""

    client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
    payload_messages = [{"content": SARVAM_PROMPT, "role": "system"}]

    if messages and len(messages) > 0:
        for m in messages:
            role = getattr(m, "type", None) or getattr(m, "role", None)
            content = getattr(m, "content", None)
            role = _normalize_role(role, m)
            payload_messages.append({"content": content, "role": role})
    else:
        payload_messages.append({"content": user_query, "role": "user"})

    response = client.chat.completions(messages=payload_messages)
    print(f"Sarvam Response : {response}")
    return ToolResponse(type=ToolResponseType.TEXT, content=str(response), response_generated_via="sarvam-m")

def _normalize_role(role: Optional[str], obj: Optional[object] = None) -> str:
    """Ensure the role is valid for Sarvam API."""
    if role in VALID_ROLES:
        return role
    if obj and hasattr(obj, "__class__"):
        clsname = obj.__class__.__name__.lower()
        return "assistant" if "ai" in clsname else "user"
    return "user"
