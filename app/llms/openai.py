from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.schemas.models import ToolResponse


def get_mini_model(temperature: float = 0.2) -> ChatOpenAI:
    """Return a GPT mini chat model for lightweight tasks."""
    mini_model = ChatOpenAI(model="gpt-5-mini", temperature=temperature)
    return mini_model

def call_openai(model_name: str, system_msg : str, human_msg: str) -> ToolResponse:
    """Call the OpenAI API."""
    model = ChatOpenAI(model=model_name, temperature=0.2).with_structured_output(schema=ToolResponse)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])