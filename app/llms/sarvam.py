import os

from sarvamai import SarvamAI

from app.schemas.models import ToolResponse
from app.schemas.enums import ToolResponseType
from app.system_prompts import SARVAM_PROMPT


def call_sarvam(user_query: str) -> ToolResponse:
    """Call the Sarvam API."""

    client = SarvamAI(
        api_subscription_key=os.getenv("SARVAM_API_KEY")
    )
    response = client.chat.completions(
        messages=[
            {"content": SARVAM_PROMPT, "role": "system"},
            {"content": user_query, "role": "user"}
            ],
    )
    print(f"Sarvam Response : {response}")
    return ToolResponse(type=ToolResponseType.TEXT, content=str(response), response_generated_via="sarvam-m")