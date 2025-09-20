from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from schemas.models import ToolResponse


def call_anthropic(model_name: str, system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Anthropic API."""
    model = ChatAnthropic(model=model_name, temperature=0.2).with_structured_output(schema=ToolResponse)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])
