from langchain_perplexity import ChatPerplexity
from langchain_core.messages import SystemMessage, HumanMessage
from models.models import ToolResponse



def call_perplexity(system_msg: str, human_msg: str) -> ToolResponse:
    """Call the Perplexity API."""
    model_name = "sonar-reasoning-pro"
    model = ChatPerplexity(model=model_name, temperature=0.2).with_structured_output(schema=ToolResponse)
    return model.invoke([SystemMessage(system_msg), HumanMessage(human_msg)])
