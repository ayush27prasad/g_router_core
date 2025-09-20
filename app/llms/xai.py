from langchain_grok import ChatGrok
from langchain_core.messages import SystemMessage, HumanMessage
from models import ToolResponse


def call_grok(system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Grok API."""
    model_name = "grok-4"
    model = ChatGrok(model=model_name, temperature=0.2).with_structured_output(schema=ToolResponse)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])