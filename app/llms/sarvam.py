from schemas.models import ToolResponse
from schemas.enums import ToolResponseType

def call_sarvam(model_name: str, system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Sarvam API."""
    # TODO : Implement the Sarvam API
    return ToolResponse(type=ToolResponseType.TEXT, content="This is a mock response from the Sarvam model")