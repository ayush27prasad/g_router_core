from typing import Optional, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from app.schemas.models import ToolResponse


def call_anthropic(model_name: str, system_msg: str, human_msg: str, messages: Optional[List[BaseMessage]] = None) -> ToolResponse:
    """Call the Anthropic API, optionally with chat history."""
    model = ChatAnthropic(model=model_name, temperature=0.2).with_structured_output(schema=ToolResponse)
    if messages and len(messages) > 0:
        full_messages: List[BaseMessage] = [SystemMessage(system_msg)] + list(messages)
        return model.invoke(full_messages)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])
