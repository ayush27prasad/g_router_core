from typing import Optional, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

from app.schemas.models import ToolResponse


def get_mini_model(temperature: float = 0.2) -> ChatOpenAI:
    """Return a GPT mini chat model for lightweight tasks."""
    mini_model_name = "gpt-5-nano"
    mini_model = ChatOpenAI(model=mini_model_name, temperature=temperature)
    return mini_model

def call_openai(model_name: str, system_msg : str, human_msg: str, messages: Optional[List[BaseMessage]] = None) -> ToolResponse:
    """Call the OpenAI API, optionally with chat history."""
    model = ChatOpenAI(model=model_name, temperature=0.2).with_structured_output(schema=ToolResponse)
    if messages and len(messages) > 0:
        full_messages: List[BaseMessage] = [SystemMessage(system_msg)] + list(messages)
        return model.invoke(full_messages)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])